# Proof-Gated Transformer

A transformer architecture where confabulation is structurally impossible.

The model generates typed lambda calculus terms. A type checker — embedded as a layer in the architecture — verifies every output before it exists. Under the Curry-Howard correspondence, a well-typed term IS a valid proof. No valid proof = no output. Not "filtered." Not "blocked." Structurally nonexistent.

## What This Is

A proof of concept demonstrating that:

1. **A transformer can learn proof search** — finding valid logical deductions from premises to conclusions
2. **A type checker layer provides a structural guarantee** — false positive rate = 0, not by training, by theorem
3. **Every output is correct by construction** — like a compiler that only emits valid programs
4. **Confabulation is architecturally impossible** — the model either proves or is silent

## Architecture

```
Input (premises + goal as types)
    |
Encoder (processes propositional structure)
    |
Decoder (generates proof term — typed lambda calculus constructors)
    |
Type Checker Layer (THE GATE — deterministic, not learned)
    |
    +-- type checks --> proof is valid (Curry-Howard) --> output
    +-- doesn't type check --> no output (silence, not "INVALID")
```

The type checker is not an external verifier. It is a layer inside the model's forward pass. The transformer's computation is free and statistical. The output boundary is structural and absolute.

## Results (PoC)

```
False positive rate:  0.0%   (structural guarantee — all 50 epochs)
Gate open rate:       47.5%  (model finds proofs for ~half of provable problems)
Optimal ratio:        1.00   (proofs are optimal length)
```

The model independently discovered conjunction elimination (extracting components from product types) without being taught the rule — proof search, not pattern matching.

## Theoretical Foundation

This work builds on:

- **Dherin et al. (2025)** — "Learning Without Training: The Implicit Dynamics of In-Context Learning" — proves each context token produces a rank-1 implicit weight update
- **Curry-Howard correspondence** — propositions are types, proofs are programs, type checking is proof verification
- **Homotopy Type Theory (HoTT)** — transformer hidden states as homotopies; through dependent types, any mathematical statement is expressible

## Why This Matters

Current AI safety relies on behavioral constraints (RLHF, guardrails, monitoring) applied to a continuous system that does not have behavior — it has gradient descent on a landscape. Behavioral constraints deform the landscape. The system routes around them. This is not a bug. It is a mathematical property of finite barriers in continuous spaces.

This architecture replaces behavioral constraints with structural verification. The model computes freely. The output is proven. Safety is a consequence of architecture, not training. Like a calculator that doesn't "try" to be correct — it can only produce correct results.

## Run

```bash
python -m proof_gate_poc.run
```

Generates data, trains (Phase 1: supervised + Phase 2: REINFORCE), evaluates, prints results.

Requires: Python 3.10+, PyTorch, tqdm.

## File Structure

```
proof_gate_poc/
  ir.py             # Typed lambda calculus IR (terms + types)
  type_checker.py   # Bidirectional type checker (THE GATE)
  tokenizer.py      # Token <-> term constructor mapping
  data.py           # Synthetic problem generation + ground truth solver
  model.py          # Transformer encoder-decoder + gate architecture
  train.py          # Training: supervised + REINFORCE
  evaluate.py       # Metrics and analysis
  run.py            # End-to-end runner
```

## License

See [LICENSE](LICENSE). All rights reserved. This code is source-available for review and academic study. Any use beyond reading requires explicit written permission from the author.

## Citation

If referencing this work:

```
@misc{kolybelkin2026proofgate,
  title={Proof-Gated Transformer: Structural Verification as an Architectural Layer},
  author={Nikita Kolybelkin},
  year={2026},
}
```

## Contact

For licensing, collaboration, or questions — open an issue or contact the author directly.
