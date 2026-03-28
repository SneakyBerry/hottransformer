"""
Interactive REPL for the proof-gated transformer.

Two modes:
  STLC mode (default): propositional logic with our Python type checker
  Agda mode (--agda):   full dependent types with Agda as frontend + gate

Usage:
    python -m proof_gate_poc.repl
    python -m proof_gate_poc.repl --agda
    python -m proof_gate_poc.repl --checkpoint path/to/model.pt

STLC input format:
    > A -> B, B -> C, A |- C
    > (A & B), B -> C |- C

Agda input format:
    > :check refl : zero ≡ zero
    > :check λ x → x : Nat → Nat
    > :dump myDefinition
    > :agda <any Agda expression to tokenize>

Commands:
    :q / :quit     — exit
    :help          — show help
    :examples      — show example inputs
    :temp <float>  — set temperature
    :tries <int>   — set number of attempts per query
    :mode stlc     — switch to STLC mode
    :mode agda     — switch to Agda mode
"""
from __future__ import annotations
import sys
import argparse
import readline  # enables arrow keys and history in input()

import torch

from .ir import Atom, Arrow, Prod, Sum, Type
from .type_checker import gate as type_check_gate
from .model import ProofGateTransformer
from .train import load_model, TrainConfig
from .tokenizer import (
    encode_problem, tokens_to_term, term_to_tokens, token_name, tokens_to_str,
    BOS, EOS, PAD,
)


def parse_type(s: str) -> tuple[Type, int]:
    """Parse a type from a string. Returns (type, next_position)."""
    s = s.lstrip()
    if not s:
        raise ValueError("Unexpected end of input")

    if s[0] == '(':
        inner, pos = parse_type(s[1:])
        rest = s[1 + pos:].lstrip()
        if rest.startswith(')'):
            result = inner
            pos = 1 + pos + (len(s[1 + pos:]) - len(rest)) + 1
            rest2 = s[pos:].lstrip()
            skip = len(s[pos:]) - len(rest2)
            if rest2.startswith('->'):
                right, rpos = parse_type(rest2[2:])
                return Arrow(result, right), pos + skip + 2 + rpos
            elif rest2.startswith('&'):
                right, rpos = parse_type(rest2[1:])
                return Prod(result, right), pos + skip + 1 + rpos
            elif rest2.startswith('|'):
                right, rpos = parse_type(rest2[1:])
                return Sum(result, right), pos + skip + 1 + rpos
            return result, pos
        else:
            raise ValueError(f"Expected ')' but got: {rest}")

    if s[0].isupper():
        result = Atom(s[0])
        rest = s[1:].lstrip()
        consumed = 1 + (len(s[1:]) - len(rest))
        if rest.startswith('->'):
            right, rpos = parse_type(rest[2:])
            return Arrow(result, right), consumed + 2 + rpos
        elif rest.startswith('&'):
            right, rpos = parse_type(rest[1:])
            return Prod(result, right), consumed + 1 + rpos
        elif rest.startswith('|'):
            right, rpos = parse_type(rest[1:])
            return Sum(result, right), consumed + 1 + rpos
        return result, consumed

    raise ValueError(f"Unexpected character: '{s[0]}' in '{s}'")


def parse_input(line: str) -> tuple[list[Type], Type]:
    """Parse 'P1, P2, ... |- Goal' into (premises, goal)."""
    if '|-' not in line:
        raise ValueError("Expected '|-' separating premises from goal. Example: A -> B, A |- B")

    premises_str, goal_str = line.split('|-', 1)

    premises = []
    if premises_str.strip():
        parts = split_by_comma(premises_str.strip())
        for part in parts:
            part = part.strip()
            if part:
                ty, _ = parse_type(part)
                premises.append(ty)

    goal, _ = parse_type(goal_str.strip())
    return premises, goal


def split_by_comma(s: str) -> list[str]:
    """Split string by commas, respecting parentheses."""
    parts = []
    depth = 0
    current = []
    for c in s:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        parts.append(''.join(current))
    return parts


STLC_EXAMPLES = """
STLC Examples:

  Simple modus ponens:
    > A -> B, A |- B

  Two-step chain:
    > A -> B, B -> C, A |- C

  Conjunction + implication:
    > (A & B), B -> C |- C

  Disjunction elimination:
    > (A | B), A -> C, B -> C |- C

  Identity:
    > |- A -> A

  Trap (unprovable):
    > A -> B, C |- B
"""

AGDA_EXAMPLES = """
Agda Examples:

  Type check a term:
    > :check refl : zero ≡ zero

  Type check with context:
    > :ctx open import Agda.Builtin.Nat
    > :check suc zero : Nat

  Check invalid (gate should close):
    > :check refl : zero ≡ suc zero

  Tokenize an expression:
    > :agda λ x → x

  Dump a definition's IR:
    > :dump myFunction
"""


# ── Natural Language Renderer ─────────────────────────────────────

def render_proof(term, premises: list = None, goal = None) -> str:
    """Render a proof term as human-readable natural language."""
    from .ir import Var, App, Lam, Pair, Fst, Snd, Inl, Inr, Case

    def premise_name(i: int) -> str:
        if premises and 0 <= i < len(premises):
            return f"посылку {i+1} ({premises[i]})"
        return f"посылку {i+1}"

    def premise_short(i: int) -> str:
        if premises and 0 <= i < len(premises):
            return f"[{i+1}] {premises[i]}"
        return f"[{i+1}]"

    def r(t, depth=0) -> str:
        indent = "  " * depth

        if isinstance(t, Var):
            return f"{indent}Берём {premise_name(t.index)}"

        elif isinstance(t, App):
            if isinstance(t.func, Var) and isinstance(t.arg, Var):
                return (f"{indent}Применяем {premise_short(t.func.index)} "
                        f"к {premise_short(t.arg.index)}")
            elif isinstance(t.func, Var):
                arg_text = r(t.arg, depth + 1)
                return (f"{indent}Применяем {premise_short(t.func.index)} к результату:\n"
                        f"{arg_text}")
            elif isinstance(t.func, App):
                func_text = r(t.func, depth + 1)
                arg_text = r(t.arg, depth + 1)
                return (f"{indent}Применяем результат:\n{func_text}\n"
                        f"{indent}к результату:\n{arg_text}")
            elif isinstance(t.func, Fst) or isinstance(t.func, Snd):
                func_text = r(t.func, depth + 1)
                arg_text = r(t.arg, depth + 1)
                return (f"{indent}Применяем:\n{func_text}\n"
                        f"{indent}к:\n{arg_text}")
            else:
                func_text = r(t.func, depth + 1)
                arg_text = r(t.arg, depth + 1)
                return (f"{indent}Применяем:\n{func_text}\n"
                        f"{indent}к:\n{arg_text}")

        elif isinstance(t, Lam):
            body_text = r(t.body, depth + 1)
            return f"{indent}Предположим посылку. Тогда:\n{body_text}"

        elif isinstance(t, Pair):
            fst_text = r(t.fst, depth + 1)
            snd_text = r(t.snd, depth + 1)
            return (f"{indent}Объединяем два доказательства:\n"
                    f"{indent}Первое:\n{fst_text}\n"
                    f"{indent}Второе:\n{snd_text}")

        elif isinstance(t, Fst):
            if isinstance(t.pair, Var):
                return f"{indent}Из {premise_name(t.pair.index)} берём первую часть"
            inner = r(t.pair, depth + 1)
            return f"{indent}Берём первую часть из:\n{inner}"

        elif isinstance(t, Snd):
            if isinstance(t.pair, Var):
                return f"{indent}Из {premise_name(t.pair.index)} берём вторую часть"
            inner = r(t.pair, depth + 1)
            return f"{indent}Берём вторую часть из:\n{inner}"

        elif isinstance(t, Inl):
            inner = r(t.term, depth + 1)
            return f"{indent}Это верно по левой альтернативе:\n{inner}"

        elif isinstance(t, Inr):
            inner = r(t.term, depth + 1)
            return f"{indent}Это верно по правой альтернативе:\n{inner}"

        elif isinstance(t, Case):
            scrut = r(t.scrutinee, depth + 1)
            left = r(t.left_branch, depth + 1)
            right = r(t.right_branch, depth + 1)
            return (f"{indent}Разбираем по случаям:\n{scrut}\n"
                    f"{indent}Случай 1 (левый):\n{left}\n"
                    f"{indent}Случай 2 (правый):\n{right}")

        return f"{indent}{t}"

    lines = []
    if premises:
        lines.append("Дано:")
        for i, p in enumerate(premises):
            lines.append(f"  [{i+1}] {p}")
        lines.append(f"Доказать: {goal}")
        lines.append("")
    lines.append("Доказательство:")
    lines.append(r(term))
    if goal:
        lines.append(f"\nСледовательно, {goal}. ∎")
    return "\n".join(lines)


def run_inference(
    model: ProofGateTransformer,
    premises: list[Type],
    goal: Type,
    device: torch.device,
    temperature: float = 0.5,
    n_tries: int = 10,
    max_len: int = 32,
) -> list[dict]:
    """Run multiple inference attempts. Returns list of valid proofs found."""
    model.eval()
    ctx = tuple(premises)
    input_tokens = encode_problem(premises, goal)

    src = torch.tensor([input_tokens], dtype=torch.long, device=device)
    results = []
    seen = set()

    with torch.no_grad():
        for attempt in range(n_tries):
            tokens, log_probs = model.generate(
                src, max_len=max_len,
                temperature=temperature,
                greedy=(attempt == 0),
            )

            seq = tokens[0].tolist()
            eos_pos = len(seq)
            for j, t in enumerate(seq):
                if t == EOS:
                    eos_pos = j
                    break
            seq = seq[:eos_pos]

            term, _ = tokens_to_term(seq)
            if term is None:
                continue

            term_str = str(term)
            if term_str in seen:
                continue
            seen.add(term_str)

            valid = type_check_gate(ctx, term, goal)
            if valid:
                results.append({
                    "term": term,
                    "tokens": seq,
                    "length": len(seq),
                    "log_prob": log_probs[0, :eos_pos].sum().item(),
                })

    results.sort(key=lambda r: r["length"])
    return results


def main():
    parser = argparse.ArgumentParser(description="Proof-Gated Transformer REPL")
    parser.add_argument("--checkpoint", default="proof_gate_poc/checkpoints/model_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--agda", action="store_true",
                        help="Start in Agda mode (use Agda frontend as tokenizer + gate)")
    args = parser.parse_args()

    print("=" * 60)
    print("PROOF-GATED TRANSFORMER — Interactive REPL")
    print("=" * 60)

    # Load model (for STLC mode)
    model = None
    config = None
    device = None
    try:
        model, config, _, _ = load_model(args.checkpoint)
        device = config.get_device()
        model.eval()
        print(f"Model loaded. Device: {device}")
    except FileNotFoundError:
        print(f"No checkpoint at {args.checkpoint} — STLC inference unavailable")
        if not args.agda:
            print("Use --agda for Agda mode, or train a model first.")

    # Agda bridge (for Agda mode)
    bridge = None
    if args.agda:
        try:
            from .agda_bridge import AgdaBridge
            bridge = AgdaBridge()
            print("Agda bridge loaded.")
        except Exception as e:
            print(f"Failed to load Agda bridge: {e}")

    mode = "agda" if args.agda and bridge else "stlc"
    temperature = 0.5
    n_tries = 10
    agda_context = ""  # persistent Agda context (imports, definitions)

    print(f"\nMode: {mode.upper()}")
    if mode == "stlc":
        print("Enter: A -> B, A |- B")
    else:
        print("Enter: :check <term> : <type>")
    print("Type :help for help, :q to quit\n")

    while True:
        try:
            prompt = f"[{mode}]> "
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # ── Global commands ──────────────────────────────────────
        if line in (":q", ":quit", ":exit"):
            print("Bye.")
            break
        elif line == ":help":
            print(__doc__)
            if mode == "stlc":
                print(STLC_EXAMPLES)
            else:
                print(AGDA_EXAMPLES)
            continue
        elif line == ":examples":
            print(STLC_EXAMPLES if mode == "stlc" else AGDA_EXAMPLES)
            continue
        elif line.startswith(":temp "):
            try:
                temperature = float(line.split()[1])
                print(f"  Temperature: {temperature}")
            except (ValueError, IndexError):
                print("  Usage: :temp <float>")
            continue
        elif line.startswith(":tries "):
            try:
                n_tries = int(line.split()[1])
                print(f"  Attempts: {n_tries}")
            except (ValueError, IndexError):
                print("  Usage: :tries <int>")
            continue
        elif line == ":mode stlc":
            mode = "stlc"
            print("  Switched to STLC mode")
            continue
        elif line == ":mode agda":
            if bridge is None:
                try:
                    from .agda_bridge import AgdaBridge
                    bridge = AgdaBridge()
                    print("  Agda bridge loaded.")
                except Exception as e:
                    print(f"  Failed to load Agda bridge: {e}")
                    continue
            mode = "agda"
            print("  Switched to Agda mode")
            continue

        # ── Agda mode commands ───────────────────────────────────
        if mode == "agda" and bridge:
            if line.startswith(":check "):
                # :check <term> : <type>
                rest = line[7:].strip()
                if " : " not in rest:
                    print("  Usage: :check <term> : <type>")
                    continue

                term_str, type_str = rest.rsplit(" : ", 1)
                print(f"  Term: {term_str}")
                print(f"  Type: {type_str}")
                print(f"  Checking with Agda...")

                valid = bridge.check_term(
                    term_str.strip(), type_str.strip(),
                    context=agda_context
                )

                if valid:
                    print(f"\n  GATE OPEN — Agda confirms: {term_str} : {type_str}")
                    print(f"  This is verified by Agda's type checker.")
                else:
                    print(f"\n  GATE CLOSED — does not type-check in Agda.")

            elif line.startswith(":ctx "):
                # Add to persistent Agda context
                ctx_line = line[5:].strip()
                agda_context += ctx_line + "\n"
                print(f"  Context added: {ctx_line}")
                print(f"  Full context:\n{agda_context}")

            elif line == ":ctx":
                if agda_context:
                    print(f"  Current Agda context:\n{agda_context}")
                else:
                    print("  No Agda context set. Use :ctx <imports/definitions>")

            elif line == ":clearctx":
                agda_context = ""
                print("  Agda context cleared.")

            elif line.startswith(":agda "):
                # Tokenize an Agda expression
                expr = line[6:].strip()
                print(f"  Tokenizing: {expr}")
                result = bridge.tokenize(expr, context=agda_context)
                if result:
                    print(f"  IR:     {result['term']}")
                    print(f"  Tokens: {result['tokens_str']}")
                else:
                    print("  Tokenization failed.")

            elif line.startswith(":dump "):
                # Dump a definition
                name = line[6:].strip()
                print(f"  Dumping: {name}")
                result = bridge.dump_definition(name, module_imports=agda_context)
                if result:
                    import json
                    print(f"  {json.dumps(result, indent=2)[:1000]}")
                else:
                    print("  Dump failed.")

            else:
                # Default: try to type-check as a complete expression
                print(f"  Use :check <term> : <type> to verify")
                print(f"  Use :agda <expr> to tokenize")
                print(f"  Use :ctx <imports> to add context")

            print()
            continue

        # ── STLC mode ────────────────────────────────────────────
        if mode == "stlc":
            if model is None:
                print("  No model loaded. Train first or use --agda mode.")
                print()
                continue

            try:
                premises, goal = parse_input(line)
            except ValueError as e:
                print(f"  Parse error: {e}")
                print()
                continue

            premises_str = ", ".join(str(p) for p in premises)
            print(f"  Premises: {premises_str}")
            print(f"  Goal:     {goal}")
            print(f"  Searching ({n_tries} attempts, temp={temperature})...")

            results = run_inference(model, premises, goal, device, temperature, n_tries)

            if results:
                best = results[0]
                print(f"\n  GATE OPEN — {len(results)} valid proof(s) found.\n")

                # Natural language explanation
                print(render_proof(best["term"], premises, goal))

                print(f"\n  Formal: {best['term']}  ({best['length']} tokens)")
                if len(results) > 1:
                    print(f"  ({len(results)-1} alternative proof(s) found)")
                print(f"  Verified by type checker. ✓")
            else:
                print(f"\n  GATE CLOSED — no valid proof found in {n_tries} attempts.")
                print(f"  Either unprovable, or model needs more search (try :tries 50)")

            print()


if __name__ == "__main__":
    main()
