# Proof-Gated Transformer

A transformer architecture where confabulation is structurally impossible.

## What This Is

A transformer that generates typed lambda calculus proof terms, verified at every step by a type checker embedded as an architectural layer. Under the Curry-Howard correspondence, a well-typed term IS a valid proof. Invalid output cannot exist — not filtered, not blocked, structurally nonexistent.

## Results

```
Gate open rate:       98.7%  (solves almost all provable problems)
False positive rate:  0.0%   (structural guarantee — by theorem, not training)
Optimality ratio:     1.00   (finds optimal-length proofs)
```

The model independently discovers logical rules not in its training data:
- Composition: `A→B, B→C ⊢ A→C` via `λ.(v2 (v1 v0))`
- K combinator: `⊢ A→(B→A)` via `λ.(λ.v1)`
- Case analysis: `(A∨B), A→C, B→C ⊢ C` via `case(v0, v1, v2)`
- Pair construction: `A, B, (A∧B)→C ⊢ C` via `(v2 ⟨v1, v0⟩)`
- Currying: `A→(B→C), A, B ⊢ C` via `((v0 v1) v2)`
- Higher-order: `(A→B)→C, A→B ⊢ C` via `(v0 (λ.(v2 v0)))`

## How It Works

```
Proof State: [?₀:Goal]              ← one hole

Step 1: type checker gives valid actions for ?₀
        model picks action via attention
        → new holes [?₁:A, ?₂:B]

Step 2: type checker gives valid actions for ?₁
        model picks → hole filled

Step 3: type checker gives valid actions for ?₂
        model picks → all holes filled

Result: complete proof term, type-checked ✓
```

At each step, only valid actions are available. The type checker constrains the search space. Value network prunes dead ends. Backtracking undoes bad choices. The model cannot produce invalid output at any step.

## Architecture

- **Unified IR**: types and terms in one typed lambda calculus (Pi, Sigma, Sum, Id, Nat, Bottom)
- **Type checker gate**: deterministic, not learned — structural guarantee
- **Hole-by-hole proving**: iterative, tactic-style — not one-shot generation
- **Pointer network**: unlimited variable references via attention to encoder
- **Value network**: predicts P(proof success | state), prunes dead branches
- **Backtracking**: undo bad choices when stuck or value drops
- **Cross-attention between holes**: holes see each other's types
- **GNN tree messages**: parent-child info flow in proof tree
- **External NTM memory**: read/write across proof steps
- **Lemma bank**: persistent, deduplicated by structure — proven theorems reusable as one-step actions
- **Agda integration**: Agda as gate (74ms/check) and tokenizer (reflected IR extraction)
- **Online self-improvement**: every valid proof updates weights — monotonic in correctness
- **Optimality reward**: from normalizer (redundant steps penalized, not proof length)

## Vocabulary

6 primitives: `APP LAM Π SET VAR ⊥`

Everything else derived as lemmas. Model discovers and stores them.

## Run

```bash
# Train 5M model: supervised (hole-by-hole) → RL → online self-improvement
python -m proof_gate_poc.train_medium

# Interactive REPL
python -m proof_gate_poc.repl

# Web UI with SSE proof streaming
python -m proof_gate_poc.server --port 8080

# Online self-improvement from checkpoint
python -m proof_gate_poc.online
```

## Project Structure

```
proof_gate_poc/
  ir.py              Unified IR (STLC + dependent types + ⊥)
  type_checker.py    THE GATE (+ structured feedback + optimality)
  tokenizer.py       Token ↔ IR (102 base + 1024 lemma slots)
  guided_search.py   Valid actions per hole (+ lemma actions)
  hole_step.py       Proof state (holes, fill, reconstruct)
  model.py           Transformer (pointer, value, cross-attn, GNN, NTM, backtracking)
  data.py            Problem generation (16 types including refutation)
  train.py           Supervised (hole-by-hole) + RL (iterative)
  online.py          Self-improvement (lemma bank, adaptive curriculum, multi-attempt)
  server.py          HTTP + SSE (hole-by-hole streaming)
  repl.py            Interactive REPL (STLC + Agda modes)
  agda_bridge.py     Agda as tokenizer/gate
  agda_server.py     Persistent Agda with warm cache
  extract_agda.py    cubical-mini/stdlib → training data
  evaluate.py        Metrics
extract/
  Extractor.agda     Reflection-based IR extraction
  DumpModule.agda    Macro for definition dumping
```

## Theoretical Foundation

- **Dherin et al. (2025)**: each context token = rank-1 implicit weight update. Current safety approach (RLHF) deforms the loss landscape — provably insufficient.
- **Curry-Howard correspondence**: propositions = types, proofs = terms, type checking = proof verification.
- **This architecture**: structural verification replaces behavioral constraints. The gate is to logical validity what softmax is to probability distributions.

## License

Source-available. See [LICENSE](LICENSE). Commercial use requires written permission.

## Citation

```
@misc{kolybelkin2026proofgate,
  title={Proof-Gated Transformer: Structural Verification as an Architectural Layer},
  author={Nikita Kolybelkin},
  year={2026},
}
```
