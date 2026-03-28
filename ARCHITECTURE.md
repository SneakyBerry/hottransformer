# Proof-Gated Transformer — Architecture

## One sentence

A transformer that fills typed holes one at a time, where a type checker makes invalid moves impossible and a value network prunes dead ends.

## How It Differs from a Standard Transformer

Standard transformer: generates tokens from a flat vocabulary. Any token can follow any token. "2+2=5" is as valid as "2+2=4". Correctness depends on weights.

This transformer: generates proof steps into typed holes. Each step is constrained by the type checker to only valid moves. A value network estimates whether the current path leads to a proof. Backtracking undoes bad choices. The output is a proof term that is correct by construction.

## Architecture

```
Input: Γ ⊢ ? : A   (premises + goal as typed hole)
  │
  ├─ Encoder: processes premise types, produces memory
  │
  ├─ Proof State: list of open holes [?₀:A, ?₁:B, ...]
  │   │
  │   ├─ Hole Selector (attention over encoded holes)
  │   │   → picks which hole to fill
  │   │
  │   ├─ Type Checker: get_valid_actions(hole.ctx, hole.goal)
  │   │   → only valid moves, invalid moves don't exist
  │   │
  │   ├─ Value Network: estimate P(proof | state) for each action
  │   │   → prune dead ends before entering them
  │   │
  │   ├─ Action Selector (attention over valid actions, weighted by value)
  │   │   → picks which action to apply
  │   │
  │   ├─ Pointer Network (for VAR): attention over encoder positions
  │   │   → points to premise, no fixed limit on context size
  │   │
  │   ├─ Fill hole → new subgoal holes appear
  │   │
  │   ├─ Value check: if value dropped → Backtrack
  │   │   → undo last choice, try next best action
  │   │
  │   └─ Repeat until no holes remain
  │
  ├─ Reconstruct term from filled holes
  │
  └─ TYPE CHECKER GATE: check(Γ, term, A)
      │
      ├─ True  → output (valid proof by Curry-Howard)
      └─ False → no output (structurally nonexistent)
```

## Components

### 1. Type Checker (the gate)

Deterministic. Not learned. Implements typing rules of STLC / Agda.

```
Γ ⊢ Var(i) : Γ(i)                    — variable lookup
Γ ⊢ App(f, x) : B  if f:A→B, x:A    — modus ponens
Γ,x:A ⊢ t : B  ⟹  Γ ⊢ Lam(t) : A→B  — → introduction
```

Returns structured feedback on failure:
- Expected type vs inferred type
- Available premises and their types
- Hint for what's needed

Soundness: if check returns True, the term inhabits the type. This is a theorem (Curry-Howard), not a training outcome.

### 2. Pointer Network (unlimited variables)

VAR tokens are not chosen from a fixed vocabulary {VAR_0...VAR_15}. The model attends over encoder positions and points to the premise it needs.

- Gate: learned scalar — use pointer (VAR) or fixed vocab (APP, LAM, etc.)
- Query: decoder hidden state projected
- Key: encoder hidden states projected
- Output: attention weight over encoder positions → selects premise

No limit on number of premises. 3 or 300 — same mechanism.

### 3. Value Network (dead-end pruning)

Second head on the model. Input: encoded proof state (pooled hole vectors). Output: P(proof completion | current state) ∈ [0, 1].

Used to:
- Rank actions before choosing (prefer actions leading to high-value states)
- Trigger backtracking when value drops below threshold
- Focus search on promising branches

Trained with outcome signal: 1.0 if proof state led to valid proof, 0.0 if not.

### 4. Backtracking

Stack of (state_snapshot, remaining_actions). When stuck or value drops:
1. Pop last choice point from stack
2. Restore state
3. Try next best action

Bounded by max_backtracks to prevent infinite loops.

### 5. Hole-by-Hole Proving (tactic mode)

Instead of generating entire proof in one shot:

```
Start:  ? : C                        (one hole)
Step 1: apply v1:B→C → ? : B         (hole filled, new hole)
Step 2: apply v0:A→B → ? : A         (hole filled, new hole)
Step 3: v2:A (direct)                 (hole filled, no new holes)
Done:   (v1 (v0 v2)) : C  ✓
```

Each step sees only valid actions for the current hole. Search space per step is tiny compared to guessing the whole proof.

### 6. Attention-Based Selection (no hardcoded choices)

| Decision | Mechanism |
|---|---|
| Which hole to fill | Attention over encoded holes |
| Which action to apply | Attention over encoded actions + value |
| Which premise to reference | Pointer network over encoder |

No if/elif dispatch. No fixed indices. Everything is learned through attention.

### 7. Shaped Reward from Type Checker Feedback

Not binary 0/1. Gradient signal based on how close the attempt was:

| Outcome | Reward | Signal |
|---|---|---|
| Valid proof, short | 1.0 | Full reward |
| Valid proof, long | 0.5-0.9 | Shorten it |
| Well-typed, wrong type | 0.05 | Close, fix the type |
| Partially valid | 0.02 | On the right track |
| Garbage | 0.0 | No signal |

### 8. Self-Improvement (persistent ΔW)

Online mode: every valid proof found during inference updates the weights.

- Gate open → supervised update (teacher-force on found proof) + REINFORCE
- Gate closed → no update (can't learn from invalid proofs — they don't exist)
- Problems generated on the fly → infinite data, no overfitting

Improvement is monotonic in correctness: only verified proofs update weights.

## Training Pipeline

```
Phase 1: Supervised (teacher forcing on solver proofs)
  → model learns proof term structure and basic patterns

Phase 2: RL (iterative, hole-by-hole, with backtracking)
  → model learns to search, value network learns to evaluate
  → fresh problems each epoch, no overfitting

Phase 3: Online (infinite self-improvement)
  → model generates + verifies + learns, forever
  → persistent ΔW from every valid proof
```

## Integration with Agda

For the PoC: Python type checker (STLC).
For production: Agda as the gate.

```python
bridge = AgdaBridge()
bridge.check_term("refl", "0 + 2 ≡ 2", ...)  # → True  (GATE OPEN)
bridge.check_term("refl", "0 + 2 ≡ 3", ...)  # → False (GATE CLOSED)
```

Agda's frontend = tokenizer (text → IR). Agda's type checker = gate (IR → valid/invalid).
Through dependent types, any mathematical statement is expressible.

## What Cannot Go Wrong

| Failure mode | Why impossible |
|---|---|
| False positive (invalid proof passes) | Type checker is a decision procedure. Proven sound. |
| Model learns to produce errors | Errors don't pass gate, don't enter training data. |
| Overfitting degrades correctness | Overfitting reduces search quality, not proof validity. |
| Adversarial input breaks guarantees | Type checker doesn't read input. It checks terms. |
| Value network misleads | Bad value = worse search, not invalid proofs. Gate catches everything. |

The gate is the invariant. Everything else can fail gracefully. The output is always correct or absent.
