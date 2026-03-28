"""
End-to-end runner.

python -m proof_gate_poc.run

Generates data, trains the proof-gated transformer, evaluates, prints results.
"""
from __future__ import annotations
import sys
import time

from .data import generate_dataset, dataset_stats
from .train import train, TrainConfig, ProofDataset, evaluate_model, collate_fn
from .evaluate import print_metrics_summary, print_proof_examples, print_gate_analysis
from .tokenizer import tokens_to_str, term_to_tokens, token_name

import torch
from torch.utils.data import DataLoader


def main():
    print("=" * 80)
    print("PROOF-GATED TRANSFORMER — PoC")
    print("Typed Lambda Calculus IR + Type Checker Gate")
    print("=" * 80)

    config = TrainConfig()

    # ── Sanity check: type checker ────────────────────────────────
    print("\n--- Type Checker Sanity Check ---")
    from .ir import Atom, Arrow, Var, App, Lam
    from .type_checker import gate, check, TypeCheckError

    A, B, C = Atom("A"), Atom("B"), Atom("C")

    # Modus ponens: A→B, A ⊢ B
    ctx = (Arrow(A, B), A)
    proof = App(Var(0), Var(1))  # apply (A→B) to A
    assert gate(ctx, proof, B), "Modus ponens should type-check"
    print(f"  ✓ App(Var(0), Var(1)) : B  [modus ponens]")

    # Chain: A→B, B→C, A ⊢ C
    ctx2 = (Arrow(A, B), Arrow(B, C), A)
    proof2 = App(Var(1), App(Var(0), Var(2)))  # apply (B→C) to (apply (A→B) to A)
    assert gate(ctx2, proof2, C), "Chain should type-check"
    print(f"  ✓ App(Var(1), App(Var(0), Var(2))) : C  [2-step chain]")

    # Invalid: try to prove C from A→B, A
    assert not gate(ctx, Var(0), C), "Var(0) : A→B ≠ C"
    assert not gate(ctx, Var(1), C), "Var(1) : A ≠ C"
    print(f"  ✓ Invalid proofs correctly rejected")

    # Lambda: ⊢ A → A (identity function)
    identity = Lam(Var(0))
    assert gate((), identity, Arrow(A, A)), "Identity should type-check"
    print(f"  ✓ Lam(Var(0)) : A → A  [identity]")

    # ── Tokenizer round-trip ──────────────────────────────────────
    print("\n--- Tokenizer Round-Trip ---")
    from .tokenizer import term_to_tokens, tokens_to_term, type_to_tokens, tokens_to_type

    for name, term in [("modus ponens", proof), ("chain", proof2), ("identity", identity)]:
        toks = term_to_tokens(term)
        recovered, _ = tokens_to_term(toks)
        assert recovered == term, f"Round-trip failed for {name}: {term} → {toks} → {recovered}"
        print(f"  ✓ {name}: {term} ↔ [{tokens_to_str(toks)}]")

    for name, ty in [("A→B", Arrow(A, B)), ("(A→B)→C", Arrow(Arrow(A, B), C))]:
        toks = type_to_tokens(ty)
        recovered, _ = tokens_to_type(toks)
        assert recovered == ty, f"Round-trip failed for {name}"
        print(f"  ✓ {name}: {ty} ↔ [{tokens_to_str(toks)}]")

    # ── Dataset ───────────────────────────────────────────────────
    print("\n--- Dataset Generation ---")
    t0 = time.time()
    test_problems = generate_dataset(5000, seed=7777)
    test_stats = dataset_stats(test_problems)
    print(f"  Test set: {test_stats}")
    print(f"  Generated in {time.time() - t0:.1f}s")

    # ── Training ──────────────────────────────────────────────────
    print("\n--- Training ---")
    model, metrics_history = train(config)

    # ── Evaluation ────────────────────────────────────────────────
    print("\n--- Final Evaluation on Test Set ---")
    device = config.get_device()
    test_dataset = ProofDataset(test_problems, max_input_len=config.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config.rl_batch_size, collate_fn=collate_fn)

    val_metrics = evaluate_model(model, test_problems, test_loader, config, device)

    # ── Results ───────────────────────────────────────────────────
    print_metrics_summary(metrics_history)
    print_proof_examples(val_metrics["examples"])
    print_gate_analysis(val_metrics)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
