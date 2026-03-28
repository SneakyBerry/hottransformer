"""
Evaluation — metrics and visualization.

Measures:
1. Gate open rate: on provable problems, how often does the model find a proof?
2. Gate false positive rate: on unprovable problems, does the gate ever open? (should be 0)
3. Average proof chain length
4. Chain length vs optimal (ratio should approach 1.0)
5. Convergence analysis
"""
from __future__ import annotations
from typing import List, Dict
from .train import EpochMetrics


def print_metrics_summary(metrics_history: List[EpochMetrics]):
    """Print a summary of training metrics."""
    if not metrics_history:
        print("No metrics to display.")
        return

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Epoch':>6} {'Loss':>8} {'Gate%':>7} {'FP%':>6} "
          f"{'AvgLen':>7} {'OptRat':>7} {'Proofs':>12} {'Time':>6}")
    print("-" * 80)

    # Print every few epochs
    step = max(1, len(metrics_history) // 20)
    for i, m in enumerate(metrics_history):
        if i % step == 0 or i == len(metrics_history) - 1:
            print(
                f"{m.epoch:6d} {m.loss:8.4f} {m.gate_open_rate:6.1%} "
                f"{m.false_positive_rate:5.1%} {m.avg_proof_length:7.1f} "
                f"{m.avg_optimal_ratio:7.2f} {m.n_valid_proofs:5d}/{m.n_total:5d} "
                f"{m.elapsed_sec:5.1f}s"
            )

    # Final stats
    final = metrics_history[-1]
    print("-" * 80)
    print(f"Final: gate_open={final.gate_open_rate:.1%}, "
          f"FP={final.false_positive_rate:.1%}, "
          f"avg_len={final.avg_proof_length:.1f}, "
          f"optimal_ratio={final.avg_optimal_ratio:.2f}")

    # Convergence analysis
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Find when gate open rate first exceeds thresholds
    for threshold in [0.01, 0.05, 0.10, 0.25, 0.50]:
        for m in metrics_history:
            if m.gate_open_rate >= threshold:
                print(f"  Gate open rate ≥ {threshold:.0%} at epoch {m.epoch}")
                break
        else:
            print(f"  Gate open rate never reached {threshold:.0%}")

    # Check if false positive rate is always 0
    fp_epochs = [m.epoch for m in metrics_history if m.false_positive_rate > 0]
    if fp_epochs:
        print(f"\n  WARNING: False positives detected at epochs: {fp_epochs}")
        print("  This should NOT happen — the type checker is deterministic.")
        print("  Investigate: likely a bug in parsing or type checking.")
    else:
        print(f"\n  ✓ False positive rate = 0 across all {len(metrics_history)} epochs")
        print("  This is the structural guarantee: the type checker cannot pass invalid proofs.")

    # Proof length trend
    lengths = [m.avg_proof_length for m in metrics_history if m.avg_proof_length > 0]
    if len(lengths) >= 2:
        first_quarter = lengths[:len(lengths)//4] if len(lengths) >= 4 else lengths[:1]
        last_quarter = lengths[-len(lengths)//4:] if len(lengths) >= 4 else lengths[-1:]
        avg_first = sum(first_quarter) / len(first_quarter)
        avg_last = sum(last_quarter) / len(last_quarter)
        if avg_first > 0:
            reduction = (avg_first - avg_last) / avg_first * 100
            print(f"\n  Proof length: {avg_first:.1f} (early) → {avg_last:.1f} (late) = {reduction:.0f}% reduction")


def print_proof_examples(examples: List[Dict]):
    """Print example proofs found by the model."""
    if not examples:
        print("\nNo valid proofs to display.")
        return

    print("\n" + "=" * 80)
    print("EXAMPLE PROOFS (gate open = valid by construction)")
    print("=" * 80)

    for i, ex in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        premises_str = ", ".join(str(p) for p in ex["premises"])
        print(f"  Premises: {premises_str}")
        print(f"  Goal:     {ex['goal']}")
        print(f"  Proof:    {ex['proof']}")
        print(f"  Length:   {ex['proof_length']} tokens (optimal: {ex['optimal_length']})")
        ratio = ex['proof_length'] / ex['optimal_length'] if ex['optimal_length'] > 0 else float('inf')
        print(f"  Ratio:    {ratio:.2f}x optimal")


def print_gate_analysis(val_metrics: Dict):
    """Analyze the gate's behavior."""
    print("\n" + "=" * 80)
    print("GATE ANALYSIS — The Structural Guarantee")
    print("=" * 80)

    print(f"\n  Provable problems:   {val_metrics['n_provable']}")
    print(f"  Unprovable problems: {val_metrics['n_unprovable']}")
    print(f"  Gate opened (valid): {val_metrics['n_valid_proofs']}")
    print(f"  False positives:     {val_metrics['n_false_positives']}")

    print(f"\n  Gate open rate:      {val_metrics['gate_open_rate']:.1%}")
    print(f"  False positive rate: {val_metrics['false_positive_rate']:.1%}")

    if val_metrics['n_false_positives'] == 0:
        print("\n  ═══════════════════════════════════════════════════")
        print("  STRUCTURAL GUARANTEE HOLDS: zero false positives.")
        print("  Every output that passed the gate is a valid proof.")
        print("  This is not learned behavior — it is a theorem.")
        print("  ═══════════════════════════════════════════════════")
    else:
        print("\n  BUG: False positives detected. The type checker")
        print("  should never pass an invalid proof. Debug needed.")
