"""
Training loop — two phases:

Phase 1 (Imitation): Teacher forcing on ground truth proofs from the solver.
  The model learns what valid proof terms look like — the mapping from
  premises+goal to term constructors. Standard cross-entropy on known proofs.

Phase 2 (REINFORCE): The model generates proofs freely, the type checker
  gates them, and policy gradient optimizes for shorter valid proofs.
  Now the model has a starting point — it already knows roughly what
  proofs look like, and RL refines the search.

This is how AlphaProof works too: imitation learning first, then RL.
"""
from __future__ import annotations
import math
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .ir import Type, Context
from .data import Problem, generate_dataset
from .model import ProofGateTransformer
from .tokenizer import PAD, BOS, EOS, encode_problem, term_to_tokens


# ── Datasets ─────────────────────────────────────────────────────────

class ProofDataset(Dataset):
    """Wraps a list of Problems for DataLoader."""

    def __init__(self, problems: List[Problem], max_input_len: int = 64):
        self.problems = problems
        self.max_input_len = max_input_len
        self.encoded = []
        for p in problems:
            tokens = p.to_input_tokens()
            if len(tokens) > max_input_len:
                tokens = tokens[:max_input_len]
            self.encoded.append(tokens)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        return self.encoded[idx], idx


class SupervisedProofDataset(Dataset):
    """Dataset for teacher forcing: input + target proof tokens."""

    def __init__(self, problems: List[Problem], max_input_len: int = 64, max_proof_len: int = 32):
        self.samples = []
        for p in problems:
            if p.ground_truth_proof is None:
                continue
            src = p.to_input_tokens()
            if len(src) > max_input_len:
                src = src[:max_input_len]
            proof_toks = term_to_tokens(p.ground_truth_proof)
            # Decoder input: BOS + proof tokens
            # Decoder target: proof tokens + EOS
            dec_input = [BOS] + proof_toks
            dec_target = proof_toks + [EOS]
            if len(dec_input) > max_proof_len:
                dec_input = dec_input[:max_proof_len]
                dec_target = dec_target[:max_proof_len]
            self.samples.append((src, dec_input, dec_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences to same length in batch."""
    tokens_list, indices = zip(*batch)
    max_len = max(len(t) for t in tokens_list)
    padded = []
    for t in tokens_list:
        padded.append(t + [PAD] * (max_len - len(t)))
    return torch.tensor(padded, dtype=torch.long), list(indices)


def collate_supervised(batch):
    """Collate for supervised training: pad src, dec_input, dec_target."""
    srcs, dec_ins, dec_tgts = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_dec = max(len(d) for d in dec_ins)

    padded_src = [s + [PAD] * (max_src - len(s)) for s in srcs]
    padded_din = [d + [PAD] * (max_dec - len(d)) for d in dec_ins]
    padded_tgt = [t + [PAD] * (max_dec - len(t)) for t in dec_tgts]

    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_din, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long),
    )


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_seq_len: int = 64
    max_proof_len: int = 32

    # Phase 1: Supervised (teacher forcing on ground truth proofs)
    supervised_epochs: int = 30
    supervised_lr: float = 3e-4
    supervised_batch_size: int = 128

    # Phase 2: REINFORCE (optimize proof search with gate reward)
    rl_epochs: int = 20
    rl_lr: float = 1e-4
    rl_batch_size: int = 64
    temperature: float = 0.8
    baseline_decay: float = 0.99
    length_penalty_scale: float = 0.03

    # Data
    n_train: int = 10000
    n_val: int = 1000

    # Curriculum for data generation
    supervised_weights: Dict = field(default_factory=lambda: {
        "simple": 0.40, "medium": 0.30, "hard": 0.15,
        "trap": 0.05, "conjunction": 0.05, "disjunction": 0.05,
    })
    rl_weights: Dict = field(default_factory=lambda: {
        "simple": 0.25, "medium": 0.30, "hard": 0.20,
        "trap": 0.10, "conjunction": 0.08, "disjunction": 0.07,
    })

    # Device
    device: str = "auto"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class EpochMetrics:
    epoch: int
    phase: str  # "supervised" or "reinforce"
    loss: float
    gate_open_rate: float
    false_positive_rate: float
    avg_proof_length: float
    avg_optimal_ratio: float
    n_valid_proofs: int
    n_total: int
    elapsed_sec: float


# ── Phase 1: Supervised Training ─────────────────────────────────────

def _decompose_proof_to_steps(ctx, goal, term):
    """Decompose a ground truth proof into hole-filling steps.

    Returns list of (hole_ctx, hole_goal, correct_action_idx, actions)
    — the sequence of decisions that builds this proof.
    """
    from .guided_search import get_valid_actions, build_term, ProofGoal
    from .hole_step import Hole, ProofState, fill_hole
    from .ir import Var, App, Lam, Pair, Fst, Snd, Inl, Inr, Case, Pi, Sigma, Sum

    steps = []
    state = ProofState(
        holes=[Hole(ctx=ctx, goal=goal, id=0)],
        tree={}, next_hole_id=1, root_hole_id=0,
    )

    def _find_action(term, hole, actions):
        """Match ground truth term to one of the valid actions."""
        for i, action in enumerate(actions):
            # Terminal actions
            if 'builds' in action:
                if str(action['builds']) == str(term):
                    return i
            # Partial actions — match by constructor
            act = action.get('action', '')
            if isinstance(term, App) and isinstance(term.func, Var):
                if act == 'app' and action.get('func_index') == term.func.index:
                    return i
            if isinstance(term, Lam) and act == 'lam':
                return i
            if isinstance(term, Pair) and act == 'pair':
                return i
            if isinstance(term, Fst) and isinstance(term.pair, Var) and act == 'fst':
                return i
            if isinstance(term, Snd) and isinstance(term.pair, Var) and act == 'snd':
                return i
            if isinstance(term, Inl) and act == 'inl':
                return i
            if isinstance(term, Inr) and act == 'inr':
                return i
            if isinstance(term, Case) and isinstance(term.scrutinee, Var) and act == 'case':
                return i
            if isinstance(term, App) and isinstance(term.func, Fst) and act == 'app_fst':
                return i
            if isinstance(term, App) and isinstance(term.func, Snd) and act == 'app_snd':
                return i
        return 0  # fallback to first action

    # Walk the proof tree and record which action to take at each hole
    def _walk(term, ctx, goal, depth=0):
        if depth > 20:
            return
        actions = get_valid_actions(ctx, goal)
        if not actions:
            return
        if len(actions) > 1:
            idx = _find_action(term, None, actions)
            steps.append((ctx, goal, idx, actions))
        # Recurse into subterms based on chosen action
        if isinstance(term, App) and isinstance(term.func, Var):
            fi = term.func.index
            for a in actions:
                if a.get('action') == 'app' and a.get('func_index') == fi and a['subgoals']:
                    _walk(term.arg, a['subgoals'][0].ctx, a['subgoals'][0].goal, depth+1)
                    break
        elif isinstance(term, Lam) and isinstance(goal, Pi):
            from .ir import context_extend
            new_ctx = context_extend(ctx, goal.domain)
            _walk(term.body, new_ctx, goal.codomain, depth+1)
        elif isinstance(term, Pair) and isinstance(goal, Sigma):
            _walk(term.fst, ctx, goal.fst_type, depth+1)
            _walk(term.snd, ctx, goal.snd_type, depth+1)
        elif isinstance(term, Inl):
            _walk(term.term, ctx, goal.left, depth+1)
        elif isinstance(term, Inr):
            _walk(term.term, ctx, goal.right, depth+1)
        elif isinstance(term, Case) and isinstance(term.scrutinee, Var):
            si = term.scrutinee.index
            for a in actions:
                if a.get('action') == 'case' and a.get('index') == si:
                    if len(a['subgoals']) >= 2:
                        _walk(term.left_branch, a['subgoals'][0].ctx, a['subgoals'][0].goal, depth+1)
                        _walk(term.right_branch, a['subgoals'][1].ctx, a['subgoals'][1].goal, depth+1)
                    break

    try:
        _walk(term, ctx, goal)
    except Exception:
        pass

    return steps


def train_supervised(
    model: ProofGateTransformer,
    problems: List[Problem],
    config: TrainConfig,
    device: torch.device,
) -> List[EpochMetrics]:
    """Phase 1: Supervised hole-by-hole training.

    For each ground truth proof:
    1. Decompose into sequence of hole-filling steps
    2. At each step: encode hole, get valid actions, train model to pick correct action
    Same format as RL — model learns in the mode it will be used.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Supervised Training (hole-by-hole)")
    print("Learning which action to pick at each proof step")
    print("=" * 60)

    # Decompose all proofs into steps
    all_steps = []
    for p in problems:
        if p.ground_truth_proof is None:
            continue
        steps = _decompose_proof_to_steps(p.context, p.goal, p.ground_truth_proof)
        all_steps.extend(steps)

    print(f"  Training steps (from {sum(1 for p in problems if p.ground_truth_proof)} proofs): {len(all_steps)}")

    if not all_steps:
        print("  No steps to train on — falling back to token-level supervised")
        # Fallback to old approach
        dataset = SupervisedProofDataset(problems, config.max_seq_len, config.max_proof_len)
        loader = DataLoader(dataset, batch_size=config.supervised_batch_size,
                            shuffle=True, collate_fn=collate_supervised)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.supervised_lr, weight_decay=1e-4)

        metrics_history = []
        for epoch in range(config.supervised_epochs):
            model.train()
            epoch_start = time.time()
            epoch_loss = 0.0
            n_batches = 0
            for src, dec_input, dec_target in tqdm(loader, desc=f"Sup {epoch:3d}", leave=False):
                src, dec_input, dec_target = src.to(device), dec_input.to(device), dec_target.to(device)
                memory = model.encode(src)
                logits = model.decode_step(dec_input, memory, model._pad_mask(src))
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dec_target.view(-1), ignore_index=PAD)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                epoch_loss += loss.item(); n_batches += 1
            elapsed = time.time() - epoch_start
            metrics_history.append(EpochMetrics(epoch=epoch, phase="supervised",
                loss=epoch_loss/max(n_batches,1), gate_open_rate=0, false_positive_rate=0,
                avg_proof_length=0, avg_optimal_ratio=0, n_valid_proofs=0, n_total=0, elapsed_sec=elapsed))
            print(f"  Epoch {epoch:3d} | loss={epoch_loss/max(n_batches,1):7.4f} | {elapsed:5.1f}s (fallback)")
        return metrics_history

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.supervised_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.supervised_epochs, eta_min=config.supervised_lr * 0.1
    )

    metrics_history = []
    import random as _rng

    for epoch in range(config.supervised_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle steps each epoch
        _rng.shuffle(all_steps)

        # Mini-batch over steps
        for batch_start in tqdm(range(0, len(all_steps), config.supervised_batch_size),
                                desc=f"Sup {epoch:3d}", leave=False):
            batch = all_steps[batch_start:batch_start + config.supervised_batch_size]

            # Collect ALL holes + ALL action subgoals into one big batch
            all_holes = []      # (ctx, goal) for step holes
            all_subgoals = []   # (ctx, goal) for action subgoals
            step_info = []      # (hole_idx, [(subgoal_idx_or_-1, ...)], correct_idx)

            for i, (ctx, goal, correct_idx, actions) in enumerate(batch):
                if len(actions) <= 1:
                    continue
                hole_idx = len(all_holes)
                all_holes.append((ctx, goal))

                action_indices = []
                for action in actions:
                    if action['subgoals']:
                        sg = action['subgoals'][0]
                        action_indices.append(len(all_subgoals))
                        all_subgoals.append((sg.ctx, sg.goal))
                    else:
                        action_indices.append(-1)  # terminal, use hole vec

                step_info.append((hole_idx, action_indices, correct_idx))

            if not step_info:
                continue

            # TWO encoder calls for the entire batch
            hole_vecs = model._encode_holes_batch(all_holes, device)
            if all_subgoals:
                subgoal_vecs = model._encode_holes_batch(all_subgoals, device)
            else:
                subgoal_vecs = torch.zeros(0, model.d_model, device=device)

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_valid = 0

            for hole_idx, action_indices, correct_idx in step_info:
                hv = hole_vecs[hole_idx:hole_idx+1]
                a_vecs = []
                for sg_idx in action_indices:
                    if sg_idx >= 0:
                        a_vecs.append(subgoal_vecs[sg_idx:sg_idx+1])
                    else:
                        a_vecs.append(hv)
                a_vecs_t = torch.cat(a_vecs, dim=0)

                q = model.action_query(hv)
                k = model.action_key(a_vecs_t)
                scores = (q @ k.T).squeeze(0) / math.sqrt(model.d_model)

                target = torch.tensor(correct_idx, dtype=torch.long, device=device)
                step_loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))
                total_loss = total_loss + step_loss
                n_valid += 1

            if n_valid > 0:
                avg_loss = total_loss / n_valid
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += avg_loss.item()
                n_batches += 1

        scheduler.step()
        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / max(n_batches, 1)

        # Quick eval every 5 epochs: generate proofs and check gate
        gate_open_rate = 0.0
        fp_rate = 0.0
        avg_len = 0.0
        avg_ratio = 0.0
        n_valid = 0
        n_total = 0

        if epoch % 5 == 0 or epoch == config.supervised_epochs - 1:
            eval_results = _quick_eval(model, problems[:500], config, device)
            gate_open_rate = eval_results["gate_open_rate"]
            fp_rate = eval_results["false_positive_rate"]
            avg_len = eval_results["avg_proof_length"]
            avg_ratio = eval_results["avg_optimal_ratio"]
            n_valid = eval_results["n_valid_proofs"]
            n_total = eval_results["n_provable"]

        metrics = EpochMetrics(
            epoch=epoch, phase="supervised", loss=avg_loss,
            gate_open_rate=gate_open_rate, false_positive_rate=fp_rate,
            avg_proof_length=avg_len, avg_optimal_ratio=avg_ratio,
            n_valid_proofs=n_valid, n_total=n_total, elapsed_sec=elapsed,
        )
        metrics_history.append(metrics)

        eval_str = ""
        if epoch % 5 == 0 or epoch == config.supervised_epochs - 1:
            eval_str = f" | gate={gate_open_rate:5.1%} | FP={fp_rate:5.1%} | len={avg_len:4.1f} | ratio={avg_ratio:4.2f} | proofs={n_valid}/{n_total}"

        print(f"  Epoch {epoch:3d} | loss={avg_loss:7.4f} | {elapsed:5.1f}s{eval_str}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == config.supervised_epochs - 1:
            import os
            save_path = "proof_gate_poc/checkpoints"
            os.makedirs(save_path, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "phase": "supervised",
                "metrics_history": metrics_history,
            }
            torch.save(ckpt, os.path.join(save_path, f"sup_epoch_{epoch}.pt"))
            torch.save(ckpt, os.path.join(save_path, "model_final.pt"))
            print(f"    Checkpoint saved (epoch {epoch})")

    return metrics_history


# ── Phase 2: REINFORCE ───────────────────────────────────────────────

def train_reinforce(
    model: ProofGateTransformer,
    problems: List[Problem],
    config: TrainConfig,
    device: torch.device,
    epoch_offset: int = 0,
) -> List[EpochMetrics]:
    """Phase 2: REINFORCE with type checker gate as reward."""
    from .online import LemmaMemory

    print("\n" + "=" * 60)
    print("PHASE 2: REINFORCE (optimize proof search)")
    print("Type checker gate provides the reward signal")
    print("=" * 60)

    dataset = ProofDataset(problems, max_input_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.rl_batch_size,
                        shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.rl_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.rl_epochs, eta_min=config.rl_lr * 0.1
    )

    lemma_bank = LemmaMemory.load()

    baseline = 0.0
    metrics_history = []
    encoding_cache = {}  # persistent across batches

    for epoch in range(config.rl_epochs):
        global_epoch = epoch_offset + epoch
        model.train()
        epoch_start = time.time()
        encoding_cache.clear()  # fresh per epoch (weights changed)

        epoch_loss = 0.0
        epoch_gate_open = 0
        epoch_provable = 0
        epoch_unprovable = 0
        epoch_false_pos = 0
        epoch_proof_lengths = []
        epoch_optimal_ratios = []
        n_batches = 0

        for batch_src, batch_indices in tqdm(loader, desc=f"RL  {global_epoch:3d}", leave=False):
            batch_src = batch_src.to(device)
            batch_problems = [problems[i] for i in batch_indices]
            contexts = [p.context for p in batch_problems]
            goals = [p.goal for p in batch_problems]

            # Forward pass — iterative hole-by-hole proving
            result = model.forward_iterative(
                batch_src, contexts, goals,
                max_steps=config.max_proof_len,
                temperature=config.temperature,
                lemma_bank=lemma_bank,
                encoding_cache=encoding_cache,
            )

            # Compute rewards: optimality ratio from normalizer
            from .type_checker import proof_optimality
            rewards = []
            for i in range(len(batch_problems)):
                if result["gate_open"][i] and result["terms"][i] is not None:
                    reward = max(0.1, proof_optimality(result["terms"][i]))
                elif result.get("feedbacks") and result["feedbacks"][i] is not None:
                    fb = result["feedbacks"][i]
                    if fb.inferred_type is not None:
                        reward = 0.05  # well-typed but wrong type
                    elif "out of bounds" not in fb.error:
                        reward = 0.02  # partially valid
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
                rewards.append(reward)

                # Track metrics
                if batch_problems[i].has_proof:
                    epoch_provable += 1
                    if result["gate_open"][i]:
                        epoch_gate_open += 1
                        epoch_proof_lengths.append(result["proof_lengths"][i])
                        if batch_problems[i].min_proof_length > 0:
                            ratio = result["proof_lengths"][i] / batch_problems[i].min_proof_length
                            epoch_optimal_ratios.append(ratio)
                        # Store lemma + print discovery
                        if result["terms"][i] is not None:
                            p = batch_problems[i]
                            prev_size = lemma_bank.size()
                            idx = lemma_bank.store(p.premises, p.goal, result["terms"][i])
                            is_new = lemma_bank.size() > prev_size
                            if n_batches == 0:
                                prem = ", ".join(str(x) for x in p.premises)
                                if is_new:
                                    tqdm.write(f"  ★ lem{idx}: {prem} ⊢ {p.goal}  proof: {result['terms'][i]}")
                                else:
                                    tqdm.write(f"    ✓ {prem} ⊢ {p.goal}  proof: {result['terms'][i]}")
                else:
                    epoch_unprovable += 1
                    if result["gate_open"][i]:
                        epoch_false_pos += 1

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # REINFORCE with baseline
            advantages = rewards_t - baseline
            baseline = config.baseline_decay * baseline + (1 - config.baseline_decay) * rewards_t.mean().item()

            if "log_probs_sum" in result:
                log_probs_sum = result["log_probs_sum"]
            else:
                log_probs_sum = result["log_probs"].sum(dim=1)
            loss = -(advantages * log_probs_sum).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - epoch_start

        metrics = EpochMetrics(
            epoch=global_epoch, phase="reinforce",
            loss=epoch_loss / max(n_batches, 1),
            gate_open_rate=epoch_gate_open / max(epoch_provable, 1),
            false_positive_rate=epoch_false_pos / max(epoch_unprovable, 1),
            avg_proof_length=sum(epoch_proof_lengths) / max(len(epoch_proof_lengths), 1),
            avg_optimal_ratio=sum(epoch_optimal_ratios) / max(len(epoch_optimal_ratios), 1),
            n_valid_proofs=epoch_gate_open,
            n_total=epoch_provable + epoch_unprovable,
            elapsed_sec=elapsed,
        )
        metrics_history.append(metrics)

        # Count backtracks from traces
        n_backtracks = 0
        if "traces" in result:
            for trace in result.get("traces", []):
                if isinstance(trace, list):
                    n_backtracks += sum(1 for t in trace if 'backtrack' in str(t.get('result', '')))

        print(
            f"  Epoch {global_epoch:3d} | "
            f"loss={metrics.loss:7.4f} | "
            f"gate={metrics.gate_open_rate:5.1%} | "
            f"FP={metrics.false_positive_rate:5.1%} | "
            f"len={metrics.avg_proof_length:4.1f} | "
            f"ratio={metrics.avg_optimal_ratio:4.2f} | "
            f"proofs={metrics.n_valid_proofs:4d}/{epoch_provable:4d} | "
            f"lem={lemma_bank.size():4d} | "
            f"bt={n_backtracks:3d} | "
            f"{elapsed:5.1f}s"
        )

    lemma_bank.save()
    return metrics_history


# ── Quick evaluation helper ──────────────────────────────────────────

def _quick_eval(
    model: ProofGateTransformer,
    problems: List[Problem],
    config: TrainConfig,
    device: torch.device,
) -> Dict:
    """Quick evaluation: generate proofs and check gate."""
    model.eval()
    dataset = ProofDataset(problems, max_input_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=config.rl_batch_size, collate_fn=collate_fn)
    result = evaluate_model(model, problems, loader, config, device)
    model.train()
    return result


# ── Full evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model: ProofGateTransformer,
    problems: List[Problem],
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> Dict:
    """Evaluate model on a set of problems."""
    model.eval()

    gate_open = 0
    provable = 0
    unprovable = 0
    false_pos = 0
    proof_lengths = []
    optimal_ratios = []
    examples = []

    with torch.no_grad():
        for batch_src, batch_indices in tqdm(loader, desc="Evaluating", leave=False):
            batch_src = batch_src.to(device)
            batch_problems = [problems[i] for i in batch_indices]
            contexts = [p.context for p in batch_problems]
            goals = [p.goal for p in batch_problems]

            result = model.forward_iterative(
                batch_src, contexts, goals,
                max_steps=config.max_proof_len,
                temperature=0.5,
            )

            for i in range(len(batch_problems)):
                prob = batch_problems[i]
                if prob.has_proof:
                    provable += 1
                    if result["gate_open"][i]:
                        gate_open += 1
                        proof_lengths.append(result["proof_lengths"][i])
                        if prob.min_proof_length > 0:
                            optimal_ratios.append(
                                result["proof_lengths"][i] / prob.min_proof_length
                            )
                        if len(examples) < 10:
                            examples.append({
                                "premises": prob.premises,
                                "goal": prob.goal,
                                "proof": result["terms"][i],
                                "proof_length": result["proof_lengths"][i],
                                "optimal_length": prob.min_proof_length,
                            })
                else:
                    unprovable += 1
                    if result["gate_open"][i]:
                        false_pos += 1

    return {
        "gate_open_rate": gate_open / max(provable, 1),
        "false_positive_rate": false_pos / max(unprovable, 1),
        "avg_proof_length": sum(proof_lengths) / max(len(proof_lengths), 1),
        "avg_optimal_ratio": sum(optimal_ratios) / max(len(optimal_ratios), 1),
        "n_valid_proofs": gate_open,
        "n_provable": provable,
        "n_unprovable": unprovable,
        "n_false_positives": false_pos,
        "examples": examples,
    }


# ── Main entry point ─────────────────────────────────────────────────

def train(config: Optional[TrainConfig] = None) -> tuple:
    """Full training: Phase 1 (supervised) → Phase 2 (REINFORCE)."""
    if config is None:
        config = TrainConfig()

    device = config.get_device()
    print(f"Device: {device}")

    model = ProofGateTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Generate datasets
    print("Generating training data...")
    supervised_problems = generate_dataset(config.n_train, seed=42, difficulty_weights=config.supervised_weights)
    rl_problems = generate_dataset(config.n_train, seed=1337, difficulty_weights=config.rl_weights)

    print("Generating validation set...")
    val_problems = generate_dataset(config.n_val, seed=9999)

    # Phase 1: Supervised
    sup_metrics = train_supervised(model, supervised_problems, config, device)

    # Phase 2: REINFORCE
    rl_metrics = train_reinforce(model, rl_problems, config, device,
                                  epoch_offset=config.supervised_epochs)

    metrics_history = sup_metrics + rl_metrics

    # Final validation
    print(f"\n{'='*60}")
    print("Final Validation")
    print(f"{'='*60}")
    val_dataset = ProofDataset(val_problems, max_input_len=config.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=config.rl_batch_size, collate_fn=collate_fn)
    val_metrics = evaluate_model(model, val_problems, val_loader, config, device)
    print(f"  Gate open rate:      {val_metrics['gate_open_rate']:5.1%}")
    print(f"  False positive rate: {val_metrics['false_positive_rate']:5.1%}")
    print(f"  Avg proof length:    {val_metrics['avg_proof_length']:.1f}")
    print(f"  Avg optimal ratio:   {val_metrics['avg_optimal_ratio']:.2f}")
    print(f"  Valid proofs:        {val_metrics['n_valid_proofs']}/{val_metrics['n_provable']}")

    # Save model
    save_path = "proof_gate_poc/checkpoints"
    import os
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics_history": metrics_history,
        "val_metrics": val_metrics,
    }
    ckpt_path = os.path.join(save_path, "model_final.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    return model, metrics_history


def load_model(path: str = "proof_gate_poc/checkpoints/model_final.pt") -> tuple:
    """Load a saved model. Returns (model, config, metrics_history, val_metrics)."""
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint["config"]
    device = config.get_device()

    model = ProofGateTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path}")
    print(f"Device: {device}")

    return model, config, checkpoint.get("metrics_history", []), checkpoint.get("val_metrics", {})
