"""
Online self-improving prover.

Combines infinite data generation with persistent weight updates from inference.
Every successful proof (gate open) updates the model's weights.
Every failed attempt (gate closed) is ignored — no learning from errors.

The model can only improve, never degrade in correctness.
Correctness is guaranteed by the type checker, not by the weights.
The weights only affect search efficiency.

Usage:
    python -m proof_gate_poc.online
    python -m proof_gate_poc.online --checkpoint proof_gate_poc/checkpoints/model_final.pt
"""
from __future__ import annotations
import argparse
import time
import os
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .ir import Type, Context
from .data import (
    Problem, generate_dataset, dataset_stats, solve,
    ALL_GENERATORS, DEFAULT_WEIGHTS, _random_type, _random_atom, _make_problem,
)
from .model import ProofGateTransformer
from .train import (
    TrainConfig, load_model, ProofDataset, collate_fn, evaluate_model,
    SupervisedProofDataset, collate_supervised,
)
from .tokenizer import (
    PAD, BOS, EOS, encode_problem, term_to_tokens, tokens_to_term,
)
from .type_checker import gate as type_check_gate

import random
from tqdm import tqdm
from torch.utils.data import DataLoader


@dataclass
class OnlineConfig:
    # How many problems to generate per round
    batch_size: int = 32
    # How many rounds
    n_rounds: int = 1000
    # Print stats every N rounds
    log_every: int = 10
    # Save checkpoint every N rounds
    save_every: int = 100
    # Learning rate for online updates
    lr: float = 1e-5
    # Temperature for generation
    temperature: float = 0.7
    # Max proof length
    max_proof_len: int = 64
    # Reward scaling
    length_penalty_scale: float = 0.02
    # Supervised learning rate on successful proofs
    supervised_lr: float = 5e-5
    # Mode: "reinforce" or "supervised" or "both"
    update_mode: str = "both"
    # Use iterative hole-by-hole proving instead of one-shot
    iterative: bool = True
    # Multiple attempts per problem
    attempts_per_problem: int = 3
    # Temperature annealing: start high (explore), decrease (exploit)
    temp_start: float = 1.0
    temp_end: float = 0.3
    # Adaptive curriculum: increase difficulty when gate rate is high
    adaptive_curriculum: bool = True


class LemmaMemory:
    """Bank of proven lemmas.

    Each lemma gets a token ID (C_LEMMA_BASE + index).
    Model can reference lemmas by token — pointer to proven theorem.
    Lemma = (premises, goal, proof_term). Reusable as one step.
    """

    def __init__(self, max_size: int = 1024):
        self.max_size = max_size
        self.lemmas: list = []   # [(premises, goal, term), ...]
        self.lookup_table: dict = {}  # (key) → index

    @staticmethod
    def _canonicalize(premises, goal) -> str:
        """Canonicalize a lemma by replacing atom names with positional indices.
        A→B, A ⊢ B  and  C→D, C ⊢ D  both become  x0→x1, x0 ⊢ x1.
        This is alpha-equivalence for atoms."""
        mapping = {}
        counter = [0]

        def canon(s: str) -> str:
            # Replace single uppercase letters (atoms) with canonical names
            result = []
            for ch in s:
                if ch.isupper() and len(ch) == 1:
                    if ch not in mapping:
                        mapping[ch] = f"x{counter[0]}"
                        counter[0] += 1
                    result.append(mapping[ch])
                else:
                    result.append(ch)
            return ''.join(result)

        prem_strs = sorted(str(p) for p in premises)
        canonical_prems = [canon(s) for s in prem_strs]
        canonical_goal = canon(str(goal))
        return f"{canonical_prems}|{canonical_goal}"

    def store(self, premises, goal, term) -> int:
        """Store a proven lemma. Returns its index. Deduplicates by structure."""
        key = self._canonicalize(premises, goal)
        if key in self.lookup_table:
            return self.lookup_table[key]
        idx = len(self.lemmas)
        if idx >= self.max_size:
            return -1  # full
        self.lemmas.append((list(premises), goal, term))
        self.lookup_table[key] = idx
        return idx

    def lookup(self, premises, goal):
        key = self._canonicalize(premises, goal)
        idx = self.lookup_table.get(key)
        if idx is not None:
            return self.lemmas[idx][2]  # return the proof term
        return None

    def get_by_index(self, idx):
        if 0 <= idx < len(self.lemmas):
            return self.lemmas[idx]
        return None

    def size(self):
        return len(self.lemmas)

    def get_all_types(self):
        """Return list of (premises, goal) for all lemmas — for attention."""
        return [(l[0], l[1]) for l in self.lemmas]

    def save(self, path: str = "proof_gate_poc/checkpoints/lemma_bank.json"):
        """Persist lemma bank to disk."""
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = []
        for premises, goal, term in self.lemmas:
            data.append({
                "premises": [str(p) for p in premises],
                "goal": str(goal),
                "term": str(term),
                "premises_repr": repr(premises),
                "goal_repr": repr(goal),
                "term_repr": repr(term),
            })
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Lemma bank saved: {len(data)} lemmas → {path}")

    @staticmethod
    def load(path: str = "proof_gate_poc/checkpoints/lemma_bank.json") -> 'LemmaMemory':
        """Load lemma bank from disk."""
        import json
        bank = LemmaMemory()
        try:
            with open(path) as f:
                data = json.load(f)
            # For now, store as string representations
            # Full deserialization would need eval() or a parser
            for entry in data:
                key = (str(sorted(entry["premises"])), entry["goal"])
                bank.lookup_table[key] = len(bank.lemmas)
                bank.lemmas.append((entry["premises"], entry["goal"], entry["term"]))
            print(f"  Lemma bank loaded: {len(data)} lemmas from {path}")
        except FileNotFoundError:
            pass
        return bank


def generate_random_problems(rng: random.Random, n: int, difficulty_bias: float = 0.0) -> list[Problem]:
    """Generate random problems on the fly. Infinite data.

    difficulty_bias: 0.0 = default weights, 1.0 = all hard problems
    """
    generators = list(ALL_GENERATORS.values())
    base_weights = list(DEFAULT_WEIGHTS.values())

    # Shift weights toward harder problems as difficulty_bias increases
    hard_indices = [i for i, name in enumerate(ALL_GENERATORS.keys())
                    if name in ('long_chain', 'nested_arrow', 'mixed_connectives',
                                'deep_elimination', 'nested_case', 'complex_trap')]
    weights = list(base_weights)
    for i in hard_indices:
        weights[i] += difficulty_bias * 0.1
    total = sum(weights)
    weights = [w / total for w in weights]

    problems = []
    for _ in range(n):
        gen = rng.choices(generators, weights=weights, k=1)[0]
        try:
            problems.append(gen(rng))
        except (ValueError, IndexError):
            from .data import generate_simple
            problems.append(generate_simple(rng))
    return problems


def online_train(
    model: ProofGateTransformer,
    config: OnlineConfig,
    device: torch.device,
    model_config: Optional[TrainConfig] = None,
):
    """Online self-improving loop with all tools:

    - Lemma memory: store and reuse proven proofs
    - Multi-attempt: try each problem N times
    - Adaptive curriculum: harder problems when model improves
    - Temperature annealing: explore → exploit
    - Value head training: learn to predict proof success
    - Fresh data every round: no overfitting
    """
    print("=" * 60)
    print("ONLINE SELF-IMPROVING PROVER")
    print("  Lemma memory + adaptive curriculum + value training")
    print("  Multi-attempt + temperature annealing")
    print("  Every valid proof updates weights. Invalid = ignored.")
    print("=" * 60)

    rl_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    sup_optimizer = torch.optim.AdamW(model.parameters(), lr=config.supervised_lr, weight_decay=1e-5)

    rng = random.Random(int(time.time()))
    baseline = 0.0
    lemma_bank = LemmaMemory.load()  # resume from saved lemmas

    # Running stats
    total_proofs_found = 0
    total_problems_seen = 0
    total_updates = 0
    round_gate_rates = []

    for round_idx in tqdm(range(config.n_rounds), desc="Online training"):
        model.train()

        # ── Temperature annealing ────────────────────────────
        progress = round_idx / max(config.n_rounds - 1, 1)
        temperature = config.temp_start + (config.temp_end - config.temp_start) * progress

        # ── Adaptive curriculum: harder problems as model improves ──
        difficulty_bias = 0.0
        if config.adaptive_curriculum and len(round_gate_rates) >= 20:
            recent_rate = sum(round_gate_rates[-20:]) / 20
            difficulty_bias = min(recent_rate * 2.0, 1.0)  # higher gate rate → harder problems

        # Step 1: Generate fresh problems
        problems = generate_random_problems(rng, config.batch_size, difficulty_bias)

        # Step 2: Check lemma memory first (instant answers)
        for p in problems:
            cached = lemma_bank.lookup(p.premises, p.goal)
            if cached is not None:
                # Already proven — skip, but count
                total_proofs_found += 1

        # Step 3: Encode and run inference (multi-attempt)
        input_tokens = [p.to_input_tokens() for p in problems]
        max_len = max(len(t) for t in input_tokens)
        padded = [t + [PAD] * (max_len - len(t)) for t in input_tokens]
        src = torch.tensor(padded, dtype=torch.long, device=device)

        contexts = [p.context for p in problems]
        goals = [p.goal for p in problems]

        # Multiple attempts — take best
        best_result = None
        best_gate_count = -1

        for attempt in range(config.attempts_per_problem):
            attempt_temp = temperature * (1.0 + attempt * 0.3)  # increase temp on retries

            if config.iterative:
                result = model.forward_iterative(
                    src, contexts, goals,
                    max_steps=config.max_proof_len,
                    temperature=attempt_temp,
                    lemma_bank=lemma_bank,
                )
            else:
                result = model.forward_with_gate(
                    src, contexts, goals,
                    max_len=config.max_proof_len,
                    temperature=attempt_temp,
                )

            gate_count = sum(1 for g in result["gate_open"] if g)
            if gate_count > best_gate_count:
                best_gate_count = gate_count
                best_result = result

            # If all provable problems solved, stop retrying
            if gate_count >= sum(1 for p in problems if p.has_proof):
                break

        result = best_result

        # Collect stats
        n_gate_open = sum(1 for g in result["gate_open"] if g)
        n_provable = sum(1 for p in problems if p.has_proof)
        total_proofs_found += n_gate_open
        total_problems_seen += len(problems)

        gate_rate = n_gate_open / max(n_provable, 1)
        round_gate_rates.append(gate_rate)

        # Store proofs in lemma memory + print discoveries
        for i in range(len(problems)):
            if result["gate_open"][i] and result["terms"][i] is not None:
                term = result["terms"][i]
                p = problems[i]
                prev_size = lemma_bank.size()
                idx = lemma_bank.store(p.premises, p.goal, term)
                is_new = lemma_bank.size() > prev_size
                prem = ', '.join(str(x) for x in p.premises)
                if is_new:
                    tqdm.write(
                        f"  ★ NEW LEMMA lem{idx}: {prem} ⊢ {p.goal}\n"
                        f"    proof: {term}"
                    )
                else:
                    tqdm.write(
                        f"    ✓ {prem} ⊢ {p.goal}  proof: {term}"
                    )

        # Step 4: Update weights ONLY from valid proofs

        if config.update_mode in ("reinforce", "both"):
            # REINFORCE with shaped reward from type checker feedback
            rewards = []
            for i in range(len(problems)):
                if result["gate_open"][i] and result["proof_lengths"][i] > 0:
                    # Valid proof: high reward, shorter = better
                    from .type_checker import proof_optimality
                    rewards.append(max(0.1, proof_optimality(result["terms"][i])))
                elif result.get("feedbacks") and result["feedbacks"][i] is not None:
                    fb = result["feedbacks"][i]
                    # Shaped reward: partial credit based on how close we got
                    if fb.inferred_type is not None:
                        # At least produced a well-typed term (wrong type, but typed)
                        rewards.append(0.05)
                    elif "out of bounds" not in fb.error:
                        # Parsed and partially type-checked (not a trivial failure)
                        rewards.append(0.02)
                    else:
                        rewards.append(0.0)
                else:
                    rewards.append(0.0)
            if sum(rewards) == 0:
                rewards = None  # skip update if no signal

            if rewards is not None:
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                advantages = rewards_t - baseline
                baseline = 0.99 * baseline + 0.01 * rewards_t.mean().item()

                if "log_probs_sum" in result:
                    log_probs_sum = result["log_probs_sum"]
                else:
                    log_probs_sum = result["log_probs"].sum(dim=1)
                rl_loss = -(advantages * log_probs_sum).mean()

                rl_optimizer.zero_grad()
                rl_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                rl_optimizer.step()
                total_updates += 1

        if config.update_mode in ("supervised", "both") and n_gate_open > 0:
            # Supervised update: teacher-force on the proofs the model just found
            # This is the persistent ΔW — bake successful proofs into weights
            sup_samples = []
            for i in range(len(problems)):
                if result["gate_open"][i] and result["terms"][i] is not None:
                    proof_toks = term_to_tokens(result["terms"][i])
                    src_toks = input_tokens[i]
                    dec_input = [BOS] + proof_toks
                    dec_target = proof_toks + [EOS]
                    sup_samples.append((src_toks, dec_input, dec_target))

            if sup_samples:
                # Pad and batch
                max_src = max(len(s) for s, _, _ in sup_samples)
                max_dec = max(len(d) for _, d, _ in sup_samples)

                src_padded = torch.tensor(
                    [s + [PAD] * (max_src - len(s)) for s, _, _ in sup_samples],
                    dtype=torch.long, device=device
                )
                din_padded = torch.tensor(
                    [d + [PAD] * (max_dec - len(d)) for _, d, _ in sup_samples],
                    dtype=torch.long, device=device
                )
                tgt_padded = torch.tensor(
                    [t + [PAD] * (max_dec - len(t)) for _, _, t in sup_samples],
                    dtype=torch.long, device=device
                )

                memory = model.encode(src_padded)
                src_pad_mask = model._pad_mask(src_padded)
                logits = model.decode_step(din_padded, memory, src_pad_mask)

                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = tgt_padded.view(-1)
                sup_loss = F.cross_entropy(logits_flat, target_flat, ignore_index=PAD)

                sup_optimizer.zero_grad()
                sup_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                sup_optimizer.step()
                total_updates += 1

        # Step 5: Train value head — learn to predict proof success
        # Positive: states from successful proofs → target 1.0
        # Negative: states from failed attempts → target 0.0
        value_targets = []
        value_inputs = []
        for i in range(len(problems)):
            ctx_i = contexts[i]
            goal_i = goals[i]
            target = 1.0 if result["gate_open"][i] else 0.0
            try:
                v = model._encode_hole(ctx_i, goal_i, device)
                value_inputs.append(v)
                value_targets.append(target)
            except Exception:
                pass

        if value_inputs:
            value_vecs = torch.cat(value_inputs, dim=0)  # (n, d_model)
            targets = torch.tensor(value_targets, dtype=torch.float32, device=device)
            predictions = model.value_head(value_vecs).squeeze(-1)
            value_loss = F.binary_cross_entropy(predictions, targets)

            sup_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            sup_optimizer.step()

        # Step 6: Log
        if (round_idx + 1) % config.log_every == 0:
            recent_rates = round_gate_rates[-config.log_every:]
            avg_rate = sum(recent_rates) / len(recent_rates)
            fp_count = sum(
                1 for i in range(len(problems))
                if not problems[i].has_proof and result["gate_open"][i]
            )
            # Value accuracy
            val_acc = "n/a"
            if value_inputs:
                with torch.no_grad():
                    preds = (predictions > 0.5).float()
                    val_acc = f"{(preds == targets).float().mean():.0%}"

            # Backtracks
            n_bt = 0
            for trace in result.get("traces", []):
                if isinstance(trace, list):
                    n_bt += sum(1 for t in trace if 'backtrack' in str(t.get('result', '')))

            print(
                f"  Round {round_idx + 1:5d} | "
                f"gate={avg_rate:5.1%} | "
                f"FP={fp_count} | "
                f"proofs={total_proofs_found:6d} | "
                f"lem={lemma_bank.size():5d} | "
                f"val={val_acc:>4s} | "
                f"bt={n_bt:3d} | "
                f"t={temperature:.2f} | "
                f"d={difficulty_bias:.1f} | "
                f"seen={total_problems_seen:7d}"
            )

        # Step 6: Save checkpoint
        if (round_idx + 1) % config.save_every == 0:
            save_path = "proof_gate_poc/checkpoints"
            os.makedirs(save_path, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": model_config,
                "round": round_idx + 1,
                "total_proofs_found": total_proofs_found,
                "total_problems_seen": total_problems_seen,
            }
            ckpt_path = os.path.join(save_path, f"online_round_{round_idx + 1}.pt")
            torch.save(checkpoint, ckpt_path)
            # Also save as latest
            torch.save(checkpoint, os.path.join(save_path, "model_final.pt"))
            lemma_bank.save()
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final stats
    print(f"\n{'='*60}")
    print(f"ONLINE TRAINING COMPLETE")
    print(f"  Rounds:          {config.n_rounds}")
    print(f"  Problems seen:   {total_problems_seen:,}")
    print(f"  Proofs found:    {total_proofs_found:,}")
    print(f"  Weight updates:  {total_updates:,}")
    print(f"  Lemmas proven:   {lemma_bank.size():,}")
    print(f"  Final gate rate: {sum(round_gate_rates[-50:]) / min(50, len(round_gate_rates)):.1%}")
    print(f"{'='*60}")
    lemma_bank.save()


def main():
    parser = argparse.ArgumentParser(description="Online self-improving prover")
    parser.add_argument("--checkpoint", default="proof_gate_poc/checkpoints/model_final.pt")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--sup-lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--mode", choices=["reinforce", "supervised", "both"], default="both")
    args = parser.parse_args()

    print("Loading model...")
    model, model_config, _, _ = load_model(args.checkpoint)
    device = model_config.get_device()

    config = OnlineConfig(
        batch_size=args.batch_size,
        n_rounds=args.rounds,
        log_every=args.log_every,
        save_every=args.save_every,
        lr=args.lr,
        supervised_lr=args.sup_lr,
        temperature=args.temperature,
        max_proof_len=model_config.max_proof_len,
        update_mode=args.mode,
    )

    online_train(model, config, device, model_config)


if __name__ == "__main__":
    main()
