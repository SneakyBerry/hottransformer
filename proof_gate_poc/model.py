"""
Proof-Gated Transformer — the architecture.

Encoder-decoder transformer where:
  - Encoder processes premises + goal (types)
  - Decoder generates proof terms (typed lambda calculus constructors)
  - Gate layer = type checker (deterministic, not learned)

The type checker is a LAYER in the forward pass. It is part of the model.
The transformer's hidden states are the homotopy from input to proof.
The type checker verifies the endpoint lands in the right fiber.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ir import Type, Term, Context
from .type_checker import gate as type_check_gate, gate_with_feedback
from .tokenizer import (
    VOCAB_SIZE, PAD, BOS, EOS,
    C_APP, C_LAM, C_PAIR, C_FST, C_SND, C_INL, C_INR, C_CASE,
    C_VAR_BASE, MAX_VARS, TERM_TOKENS,
    tokens_to_term, term_to_tokens, token_name,
    encode_problem,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ProofGateTransformer(nn.Module):
    """
    The full architecture:
      Input tokens → Encoder → Decoder → Term constructor logits (= IR) → Type checker gate

    The output logits are over term constructor tokens.
    The type checker operates on the decoded term.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Shared embedding (input types + output term constructors use same vocab)
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.scale = math.sqrt(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output projection to term constructor vocabulary (without VAR tokens)
        self.output_proj = nn.Linear(d_model, VOCAB_SIZE)

        # Pointer network: for VAR, point to encoder position
        self.pointer_gate = nn.Linear(d_model, 1)
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)

        # Hole selector: attention over encoded holes to pick which to fill
        self.hole_query = nn.Linear(d_model, d_model)
        self.hole_key = nn.Linear(d_model, d_model)

        # Action scorer: attention over encoded actions to pick which to apply
        self.action_query = nn.Linear(d_model, d_model)
        self.action_key = nn.Linear(d_model, d_model)

        # Encode a hole (ctx + goal) into a single vector
        self.hole_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Cross-attention between holes: holes see each other's types
        # "if I solve ?₁ this way, ?₂ becomes easier"
        self.hole_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.hole_cross_norm = nn.LayerNorm(d_model)

        # GNN-style message passing on proof tree structure
        # Parent-child messages between holes and filled nodes
        self.tree_message = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.tree_update = nn.GRUCell(d_model, d_model)

        # External read/write memory (Neural Turing Machine style)
        # Stores intermediate proof insights across steps
        self.memory_size = 32
        self.memory_read_query = nn.Linear(d_model, d_model)
        self.memory_read_key = nn.Linear(d_model, d_model)
        self.memory_write_query = nn.Linear(d_model, d_model)
        self.memory_write_key = nn.Linear(d_model, d_model)
        self.memory_erase = nn.Linear(d_model, d_model)
        self.memory_add = nn.Linear(d_model, d_model)

        # Value network: predicts probability of reaching a valid proof from current state
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # state + memory context
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Mask: restrict output to term constructor tokens + EOS only
        self.register_buffer('output_mask', self._build_output_mask())

    def _build_output_mask(self) -> torch.Tensor:
        """Create a mask that allows only term constructor tokens and EOS in output."""
        mask = torch.full((VOCAB_SIZE,), float('-inf'))
        mask[EOS] = 0.0
        for tok in TERM_TOKENS:
            mask[tok] = 0.0
        return mask

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(size, device=device)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens == PAD  # (batch, seq_len)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode input (premises + goal)."""
        src_mask = self._pad_mask(src)
        x = self.pos_encoder(self.embedding(src) * self.scale)
        return self.encoder(x, src_key_padding_mask=src_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode one step: returns logits over term constructors + pointer to encoder positions.

        For non-VAR tokens: standard vocabulary logits.
        For VAR tokens: pointer attention over encoder positions.
        Gate learns when to point vs use vocab.

        Output shape: (batch, seq_len, VOCAB_SIZE + max_encoder_len)
        First VOCAB_SIZE dims = fixed vocabulary logits
        Remaining dims = pointer logits to encoder positions (mapped to VAR_i)
        """
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        x = self.pos_encoder(self.embedding(tgt) * self.scale)
        x = self.decoder(x, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_pad_mask)

        # Fixed vocabulary logits (constructors: APP, LAM, PAIR, etc.)
        vocab_logits = self.output_proj(x)
        vocab_logits = vocab_logits + self.output_mask

        # Pointer logits: attention from decoder to encoder positions
        # query: decoder hidden states, key: encoder hidden states
        ptr_query = self.pointer_query(x)      # (batch, dec_len, d_model)
        ptr_key = self.pointer_key(memory)      # (batch, enc_len, d_model)
        ptr_logits = torch.bmm(ptr_query, ptr_key.transpose(1, 2)) / math.sqrt(self.d_model)
        # (batch, dec_len, enc_len)

        # Mask padded encoder positions
        if src_pad_mask is not None:
            ptr_logits = ptr_logits.masked_fill(src_pad_mask.unsqueeze(1), float('-inf'))

        # Gate: probability of using pointer (for VAR) vs fixed vocab
        gate_prob = torch.sigmoid(self.pointer_gate(x))  # (batch, dec_len, 1)

        # Suppress VAR tokens in fixed vocab — pointer handles them
        vocab_logits_no_var = vocab_logits.clone()
        for vi in range(MAX_VARS):
            vocab_logits_no_var[:, :, C_VAR_BASE + vi] = float('-inf')

        # Combine: vocab logits weighted by (1 - gate), pointer logits weighted by gate
        # We map pointer positions back to VAR_i tokens for compatibility
        # Encoder position i → VAR_i (premise at position i in context)
        enc_len = memory.size(1)
        combined = torch.full(
            (x.size(0), x.size(1), VOCAB_SIZE + enc_len),
            float('-inf'), device=x.device
        )

        # Fixed vocab part (non-VAR constructors)
        combined[:, :, :VOCAB_SIZE] = vocab_logits_no_var + torch.log(1 - gate_prob + 1e-8)

        # Pointer part — mapped to positions after VOCAB_SIZE
        combined[:, :, VOCAB_SIZE:] = ptr_logits + torch.log(gate_prob + 1e-8)

        # For backward compatibility: collapse back to VOCAB_SIZE
        # Map pointer positions to VAR tokens
        final_logits = vocab_logits_no_var.clone()
        for pos in range(min(enc_len, MAX_VARS)):
            # Add pointer probability for this position to VAR_{pos}
            final_logits[:, :, C_VAR_BASE + pos] = (
                torch.log(gate_prob.squeeze(-1) + 1e-8) + ptr_logits[:, :, pos]
            )

        return final_logits

    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 32,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively generate proof term tokens.

        Returns:
            tokens: (batch, seq_len) generated token IDs
            log_probs: (batch, seq_len) log probability of each token
        """
        batch_size = src.size(0)
        device = src.device

        memory = self.encode(src)
        src_pad_mask = self._pad_mask(src)

        # Start with BOS
        generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
        all_log_probs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decode_step(generated, memory, src_pad_mask)
            next_logits = logits[:, -1, :] / temperature  # (batch, vocab)

            probs = F.softmax(next_logits, dim=-1)
            if greedy:
                next_token = probs.argmax(dim=-1)
            else:
                next_token = torch.multinomial(probs, 1).squeeze(-1)

            log_prob = F.log_softmax(next_logits, dim=-1)
            token_log_prob = log_prob.gather(1, next_token.unsqueeze(1)).squeeze(1)

            # Mask finished sequences
            next_token = next_token.masked_fill(finished, PAD)
            token_log_prob = token_log_prob.masked_fill(finished, 0.0)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            all_log_probs.append(token_log_prob)

            finished = finished | (next_token == EOS)
            if finished.all():
                break

        log_probs = torch.stack(all_log_probs, dim=1)  # (batch, gen_len)
        return generated[:, 1:], log_probs  # skip BOS

    def forward_with_gate(
        self,
        src: torch.Tensor,
        contexts: List[Context],
        goals: List[Type],
        max_len: int = 32,
        temperature: float = 1.0,
    ) -> dict:
        """Full forward pass including the type checker gate.

        This is the complete architecture in one call:
        Input → Encoder → Decoder → Term constructors (IR) → Type checker → Gate

        Returns dict with:
            tokens: generated token sequences
            log_probs: log probabilities
            terms: parsed Term objects (None if parse failed)
            gate_open: boolean for each item (True = valid proof)
            proof_lengths: number of tokens in each proof (0 if gate closed)
        """
        tokens, log_probs = self.generate(src, max_len, temperature)

        batch_size = tokens.size(0)
        terms = []
        gate_results = []
        proof_lengths = []
        feedbacks = []

        for i in range(batch_size):
            # Extract token sequence (strip padding and EOS)
            seq = tokens[i].tolist()
            # Find EOS
            eos_pos = len(seq)
            for j, t in enumerate(seq):
                if t == EOS:
                    eos_pos = j
                    break
            seq = seq[:eos_pos]

            # Parse tokens to term (this IS reading the IR)
            term, _ = tokens_to_term(seq)
            terms.append(term)

            if term is not None:
                # TYPE CHECKER GATE — with structured feedback
                fb = gate_with_feedback(contexts[i], term, goals[i])
                gate_results.append(fb.valid)
                proof_lengths.append(len(seq) if fb.valid else 0)
                feedbacks.append(fb)
            else:
                gate_results.append(False)
                proof_lengths.append(0)
                feedbacks.append(None)

        return {
            "tokens": tokens,
            "log_probs": log_probs,
            "terms": terms,
            "gate_open": gate_results,
            "proof_lengths": proof_lengths,
            "feedbacks": feedbacks,
        }

    def _encode_hole(self, hole_ctx, hole_goal, device):
        """Encode a single hole. For multiple holes use _encode_holes_batch."""
        return self._encode_holes_batch([(hole_ctx, hole_goal)], device)

    def _encode_holes_batch(self, hole_list, device):
        """Batch-encode multiple holes in ONE encoder forward pass.

        hole_list: list of (ctx, goal) tuples
        Returns: (n_holes, d_model) tensor
        """
        if not hole_list:
            return torch.zeros(0, self.d_model, device=device)

        # Tokenize all holes
        all_tokens = []
        for ctx, goal in hole_list:
            tokens = encode_problem(list(ctx) if not isinstance(ctx, list) else ctx, goal)
            all_tokens.append(tokens)

        # Pad to same length
        max_len = max(len(t) for t in all_tokens)
        padded = [t + [PAD] * (max_len - len(t)) for t in all_tokens]
        src = torch.tensor(padded, dtype=torch.long, device=device)

        # ONE encoder forward pass for all holes
        memory = self.encode(src)  # (n_holes, max_len, d_model)
        mask = self._pad_mask(src)  # (n_holes, max_len)
        lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = (memory * (~mask).unsqueeze(-1).float()).sum(dim=1) / lengths  # (n_holes, d_model)

        return self.hole_encoder(pooled)  # (n_holes, d_model)

    def _encode_holes_with_cross_attn(self, holes, device):
        """Encode all holes in one batch + cross-attention."""
        if not holes:
            return torch.zeros(0, self.d_model, device=device)

        # Batch encode all holes at once
        hole_list = [(h.ctx, h.goal) for h in holes]
        hole_vecs = self._encode_holes_batch(hole_list, device)

        if len(holes) <= 1:
            return hole_vecs

        # Cross-attention: holes see each other
        hv = hole_vecs.unsqueeze(0)
        attended, _ = self.hole_cross_attn(hv, hv, hv)
        attended = self.hole_cross_norm(hv + attended)
        return attended.squeeze(0)

    def _tree_message_pass(self, hole_vecs, parent_vec, device):
        """GNN message passing: parent node sends message to child holes."""
        if hole_vecs.size(0) == 0 or parent_vec is None:
            return hole_vecs
        # Concatenate parent with each hole, compute message, update via GRU
        parent_expanded = parent_vec.expand(hole_vecs.size(0), -1)
        messages = self.tree_message(torch.cat([hole_vecs, parent_expanded], dim=-1))
        updated = self.tree_update(messages, hole_vecs)
        return updated

    def _init_memory(self, device):
        """Initialize external memory bank."""
        return torch.zeros(1, self.memory_size, self.d_model, device=device)

    def _memory_read(self, query_vec, memory):
        """Read from external memory via attention."""
        q = self.memory_read_query(query_vec).unsqueeze(1)  # (1, 1, d)
        k = self.memory_read_key(memory)  # (1, M, d)
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        read = torch.bmm(weights, memory).squeeze(1)  # (1, d)
        return read

    def _memory_write(self, write_vec, memory):
        """Write to external memory: erase old + add new."""
        q = self.memory_write_query(write_vec).unsqueeze(1)
        k = self.memory_write_key(memory)
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_model), dim=-1)
        # Erase
        erase = torch.sigmoid(self.memory_erase(write_vec)).unsqueeze(1)  # (1, 1, d)
        memory = memory * (1 - torch.bmm(weights.transpose(1, 2), erase))
        # Add
        add = self.memory_add(write_vec).unsqueeze(1)  # (1, 1, d)
        memory = memory + torch.bmm(weights.transpose(1, 2), add)
        return memory

    def _estimate_value(self, state_holes, device, memory=None) -> float:
        """Value network with memory context."""
        if not state_holes:
            return 1.0
        hole_list = [(h.ctx, h.goal) for h in state_holes]
        hole_vecs = self._encode_holes_batch(hole_list, device)
        state_vec = hole_vecs.mean(dim=0, keepdim=True)

        if memory is not None:
            mem_context = self._memory_read(state_vec, memory)
            combined = torch.cat([state_vec, mem_context], dim=-1)
        else:
            combined = torch.cat([state_vec, state_vec], dim=-1)

        return self.value_head(combined).item()

    def forward_iterative(
        self,
        src: torch.Tensor,
        contexts: List[Context],
        goals: List[Type],
        max_steps: int = 20,
        temperature: float = 1.0,
        max_backtracks: int = 5,
        lemma_bank=None,
        encoding_cache: dict = None,
    ) -> dict:
        """Hole-by-hole proof construction with:
        - Attention-based hole selection
        - Attention-based action selection
        - Value network to evaluate proof states
        - Backtracking when stuck or value drops

        At each step:
        1. Value network estimates chance of success
        2. If value too low → backtrack to last choice point
        3. Attention selects which hole to fill
        4. Type checker gives valid actions
        5. Attention selects action, value network ranks candidates
        6. Fill hole, push state to backtrack stack
        7. Repeat until no holes or max backtracks exhausted
        """
        from .guided_search import get_valid_actions, build_term
        from .hole_step import Hole, ProofState, fill_hole, reconstruct_term
        import copy

        batch_size = src.size(0)
        device = src.device
        global_memory = self.encode(src)

        # Hole encoding cache — detach tensors to avoid backward-through-graph issues
        _hole_cache = encoding_cache if encoding_cache is not None else {}
        _use_cache = not self.training  # only cache in eval mode (no grad needed)

        def encode_hole_cached(ctx, goal):
            key = (id(ctx), str(goal))
            if _use_cache and key in _hole_cache:
                return _hole_cache[key]
            result = self._encode_holes_batch([(ctx, goal)], device)
            if _use_cache:
                _hole_cache[key] = result.detach()
            return result

        def encode_holes_cached(holes):
            """Batch encode, using cache for already-seen holes (eval only)."""
            uncached = []
            uncached_idx = []
            results = [None] * len(holes)

            for i, h in enumerate(holes):
                key = (id(h.ctx), str(h.goal))
                if _use_cache and key in _hole_cache:
                    results[i] = _hole_cache[key]
                else:
                    uncached.append((h.ctx, h.goal))
                    uncached_idx.append(i)

            if uncached:
                encoded = self._encode_holes_batch(uncached, device)
                for j, idx in enumerate(uncached_idx):
                    vec = encoded[j:j+1]
                    results[idx] = vec
                    if _use_cache:
                        h = holes[idx]
                        _hole_cache[(id(h.ctx), str(h.goal))] = vec.detach()

            stacked = torch.cat([r for r in results if r is not None], dim=0)

            # Cross-attention if >1
            if len(holes) > 1:
                hv = stacked.unsqueeze(0)
                attended, _ = self.hole_cross_attn(hv, hv, hv)
                stacked = self.hole_cross_norm(hv + attended).squeeze(0)

            return stacked

        all_results = []

        for b in range(batch_size):
            ctx = contexts[b]
            goal = goals[b]
            state = ProofState(
                holes=[Hole(ctx=ctx, goal=goal, id=0)],
                tree={}, next_hole_id=1, root_hole_id=0,
            )

            log_prob_parts = []
            n_steps = 0
            n_backtracks = 0
            trace = []
            b_memory = global_memory[b:b+1]
            ext_memory = self._init_memory(device)
            last_hole_vec = None
            backtrack_stack = []

            for step in range(max_steps):
                if not state.holes:
                    break

                n_steps += 1

                # ── Encode holes (cached + cross-attn) ───────────
                hole_vecs = encode_holes_cached(state.holes)

                if last_hole_vec is not None and hole_vecs.size(0) > 0:
                    hole_vecs = self._tree_message_pass(hole_vecs, last_hole_vec, device)

                state_vec = hole_vecs.mean(dim=0, keepdim=True)
                mem_context = self._memory_read(state_vec, ext_memory)

                # ── Select hole ──────────────────────────────────
                if len(state.holes) == 1:
                    hole_idx = 0
                else:
                    global_vec = b_memory.mean(dim=1) + mem_context
                    q = self.hole_query(global_vec)
                    k = self.hole_key(hole_vecs)
                    scores = (q @ k.T).squeeze(0) / math.sqrt(self.d_model) / temperature
                    hole_probs = F.softmax(scores, dim=-1)
                    hole_idx = torch.multinomial(hole_probs, 1).item()
                    log_prob_parts.append(F.log_softmax(scores, dim=-1)[hole_idx])

                hole = state.holes[hole_idx]
                hole_vec = hole_vecs[hole_idx:hole_idx+1]
                actions = get_valid_actions(hole.ctx, hole.goal, lemma_bank=lemma_bank)

                # ── Stuck → backtrack ────────────────────────────
                if not actions:
                    if backtrack_stack and n_backtracks < max_backtracks:
                        saved = backtrack_stack.pop()
                        state, ext_memory = saved[0], saved[1]
                        n_backtracks += 1
                        trace.append({'goal': str(hole.goal), 'result': 'backtrack'})
                        continue
                    trace.append({'goal': str(hole.goal), 'result': 'stuck'})
                    break

                # ── Early exit: terminal action (no subgoals) ────
                terminal = [a for a in actions if not a['subgoals']]
                if terminal:
                    chosen = terminal[0]
                    last_hole_vec = hole_vec
                    ext_memory = self._memory_write(hole_vec, ext_memory)
                    state = fill_hole(state, hole_idx, chosen)
                    trace.append({'goal': str(hole.goal), 'forced': True,
                                  'action': chosen['description']})
                    continue

                # ── Single action: forced ────────────────────────
                if len(actions) == 1:
                    last_hole_vec = hole_vec
                    ext_memory = self._memory_write(hole_vec, ext_memory)
                    state = fill_hole(state, hole_idx, actions[0])
                    trace.append({'goal': str(hole.goal), 'forced': True,
                                  'action': actions[0]['description']})
                    continue

                # ── Multiple actions: batch-encode subgoals ──────
                action_hole_list = []
                action_hole_map = []
                for ai, action in enumerate(actions):
                    if action['subgoals']:
                        sg = action['subgoals'][0]
                        action_hole_list.append((sg.ctx, sg.goal))
                        action_hole_map.append(len(action_hole_list) - 1)
                    else:
                        action_hole_map.append(-1)

                if action_hole_list:
                    encoded_actions = self._encode_holes_batch(action_hole_list, device)
                else:
                    encoded_actions = torch.zeros(0, self.d_model, device=device)

                action_vecs = []
                action_values = []
                for ai, action in enumerate(actions):
                    if action_hole_map[ai] >= 0:
                        av = encoded_actions[action_hole_map[ai]:action_hole_map[ai]+1]
                    else:
                        av = hole_vec
                    action_vecs.append(av)

                    # Lazy value: skip for 2 actions, just use attention
                    if len(actions) <= 2:
                        action_values.append(0.5)
                    else:
                        future_holes = list(state.holes[hole_idx+1:]) + list(state.holes[:hole_idx])
                        for sg in action['subgoals']:
                            future_holes.append(Hole(ctx=sg.ctx, goal=sg.goal, id=-1))
                        value = self._estimate_value(future_holes, device, ext_memory) if future_holes else 1.0
                        action_values.append(value)

                action_vecs_t = torch.cat(action_vecs, dim=0)

                q = self.action_query(hole_vec + mem_context)
                k = self.action_key(action_vecs_t)
                attn_scores = (q @ k.T).squeeze(0) / math.sqrt(self.d_model)
                value_scores = torch.tensor(action_values, dtype=torch.float32, device=device)
                combined = (attn_scores + value_scores * 2.0) / temperature

                probs = F.softmax(combined, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
                log_prob_parts.append(F.log_softmax(combined, dim=-1)[action_idx])

                chosen = actions[action_idx]

                if len(actions) > 1:
                    backtrack_stack.append((copy.deepcopy(state), ext_memory.clone()))

                last_hole_vec = hole_vec
                ext_memory = self._memory_write(hole_vec, ext_memory)

                trace.append({
                    'goal': str(hole.goal), 'forced': False,
                    'action': chosen['description'],
                    'n_options': len(actions),
                    'value': action_values[action_idx],
                })
                state = fill_hole(state, hole_idx, chosen)

                # ── Value check → backtrack ──────────────────────
                if state.holes and len(actions) > 2:
                    current_value = self._estimate_value(state.holes, device, ext_memory)
                    if current_value < 0.05 and backtrack_stack and n_backtracks < max_backtracks:
                        saved = backtrack_stack.pop()
                        state, ext_memory = saved[0], saved[1]
                        n_backtracks += 1
                        trace.append({'result': 'value_backtrack', 'value': current_value})

            term = reconstruct_term(state)
            valid = type_check_gate(ctx, term, goal) if term is not None else False

            all_results.append({
                'term': term,
                'valid': valid,
                'steps': n_steps,
                'backtracks': n_backtracks,
                'remaining_holes': len(state.holes),
                'trace': trace,
                'log_prob_tensor': torch.stack(log_prob_parts).sum() if log_prob_parts else torch.tensor(0.0, device=device, requires_grad=True),
            })

        return {
            "terms": [r['term'] for r in all_results],
            "gate_open": [r['valid'] for r in all_results],
            "proof_lengths": [r['steps'] if r['valid'] else 0 for r in all_results],
            "log_probs_sum": torch.stack([r['log_prob_tensor'] for r in all_results]),
            "traces": [r['trace'] for r in all_results],
            "feedbacks": [None] * batch_size,
        }
