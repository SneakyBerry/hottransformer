"""
Agda → Training Data extraction pipeline.

Converts JSON output from our Agda reflection extractor into
training samples for the dependent type transformer.

JSON format from Extractor.agda uses {"tag": "var", "index": ..., "args": [...]}
matching Agda.Builtin.Reflection.Term constructors.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

from .ir import (
    Term, Var, App, Lam, Pi, Sigma, Pair, Fst, Snd,
    Universe, Set, Prop, Id, Refl, J, Nat, Zero, Suc, NatElim,
)
from .tokenizer import term_to_tokens, type_to_tokens, tokens_to_str


def reflected_to_term(data) -> Optional[Term]:
    """Convert Agda reflected Term JSON to our IR.

    Format from Extractor.agda:
      {"tag": "var", "index": 0, "args": [...]}
      {"tag": "con", "name": "...", "args": [...]}
      {"tag": "def", "name": "...", "args": [...]}
      {"tag": "lam", "visibility": "...", "body": {"name": "...", "body": ...}}
      {"tag": "pi", "domain": {"visibility": "...", "term": ...}, "codomain": {"name": "...", "body": ...}}
      {"tag": "sort", "sort": {"kind": "set", "level": ...}}
      {"tag": "lit", "value": "..."}
      {"tag": "pat-lam", "clauses": [...], "args": [...]}
      {"tag": "meta", "args": [...]}
      {"tag": "unknown"}
    """
    if data is None or not isinstance(data, dict):
        return None

    tag = data.get("tag")

    if tag == "var":
        idx = data.get("index", 0)
        if isinstance(idx, str):
            idx = int(idx) if idx.isdigit() else 0
        term = Var(idx)
        for arg in data.get("args", []):
            if isinstance(arg, dict):
                a = reflected_to_term(arg.get("term", arg))
                if a is not None:
                    term = App(term, a)
        return term

    elif tag == "lam":
        body_data = data.get("body", {})
        if isinstance(body_data, dict):
            body = reflected_to_term(body_data.get("body", body_data))
        else:
            body = None
        return Lam(body) if body else None

    elif tag == "pi":
        domain_data = data.get("domain", {})
        codomain_data = data.get("codomain", {})
        domain = reflected_to_term(domain_data.get("term", domain_data))
        codomain = reflected_to_term(codomain_data.get("body", codomain_data))
        if domain and codomain:
            return Pi(domain, codomain)
        return None

    elif tag == "sort":
        sort_data = data.get("sort", {})
        kind = sort_data.get("kind", "set")
        level_data = sort_data.get("level")
        level = 0
        if isinstance(level_data, (int, float)):
            level = int(level_data)
        elif isinstance(level_data, str) and level_data.isdigit():
            level = int(level_data)
        # level_data could be a term (variable level) — default to 0
        if "set" in kind:
            return Universe(Set(level))
        elif "prop" in kind:
            return Universe(Prop(level))
        return Universe(Set(0))

    elif tag == "con":
        name = data.get("name", "")
        args = []
        for a in data.get("args", []):
            if isinstance(a, dict):
                t = reflected_to_term(a.get("term", a))
                if t is not None:
                    args.append(t)

        # Map known constructors
        if "refl" in name.lower():
            return Refl()
        elif name.endswith(".zero") or name == "zero":
            return Zero()
        elif name.endswith(".suc") or name == "suc":
            return Suc(args[0]) if args else None
        elif "," in name or "pair" in name.lower() or "mk" in name.lower():
            if len(args) >= 2:
                return Pair(args[0], args[1])
            elif len(args) == 1:
                return args[0]

        # Generic constructor: return as application chain
        if args:
            term = Var(0)  # placeholder for constructor reference
            for a in args:
                term = App(term, a)
            return term
        return Var(0)  # nullary constructor placeholder

    elif tag == "def":
        name = data.get("name", "")
        args = []
        for a in data.get("args", []):
            if isinstance(a, dict):
                t = reflected_to_term(a.get("term", a))
                if t is not None:
                    args.append(t)

        # Map known definitions to our IR
        if "Agda.Builtin.Nat.Nat" in name or "ℕ" in name:
            return Nat()
        elif "Sigma" in name or "Σ" in name:
            if len(args) >= 2:
                return Sigma(args[0], args[1])
        elif "≡" in name or "＝" in name or ".Id" in name:
            if len(args) >= 3:
                return Id(args[0], args[1], args[2])
            elif len(args) >= 2:
                return Id(Var(0), args[0], args[1])
        elif "Agda.Primitive.Level" in name:
            return Universe(Set(0))  # treat Level as Set for now

        # Generic definition: application chain
        if not args:
            return Var(0)
        term = Var(0)
        for a in args:
            term = App(term, a)
        return term

    elif tag == "lit":
        return Zero()  # simplified

    elif tag == "pat-lam":
        # Pattern-matching lambda — extract first clause body
        clauses = data.get("clauses", [])
        if clauses and isinstance(clauses[0], dict):
            body = clauses[0].get("body")
            if body:
                return reflected_to_term(body)
        return None

    elif tag == "meta" or tag == "unknown":
        return None

    return None


def reflected_body_to_term(body_data) -> Optional[Term]:
    """Convert body (list of clauses or single term) to IR."""
    if isinstance(body_data, list):
        if not body_data:
            return None
        clause = body_data[0]
        if isinstance(clause, dict):
            body = clause.get("body")
            if body:
                return reflected_to_term(body)
    elif isinstance(body_data, dict):
        tag = body_data.get("tag")
        if tag:
            return reflected_to_term(body_data)
        # Could be {"kind": "data"} etc — skip non-function definitions
        kind = body_data.get("kind")
        if kind in ("data", "record", "data-con", "axiom", "prim"):
            return None
    return None


def extract_training_data(dump_path: str) -> List[Dict]:
    """Read extracted Agda definitions and produce training samples."""
    with open(dump_path, 'r') as f:
        definitions = json.load(f)

    samples = []
    n_type_ok = 0
    n_body_ok = 0

    for defn in definitions:
        name = defn.get("name", "?")
        ty_data = defn.get("type")
        body_data = defn.get("body")

        if ty_data is None or body_data is None:
            continue

        ty = reflected_to_term(ty_data)
        if ty is not None:
            n_type_ok += 1

        body = reflected_body_to_term(body_data)
        if body is not None:
            n_body_ok += 1

        if ty is None or body is None:
            continue

        try:
            input_toks = type_to_tokens(ty)
            target_toks = term_to_tokens(body)
        except (ValueError, Exception):
            continue

        samples.append({
            "name": name,
            "input_tokens": input_toks,
            "target_tokens": target_toks,
            "type": str(ty),
            "term": str(body),
        })

    print(f"  Types parsed: {n_type_ok}/{len(definitions)}")
    print(f"  Bodies parsed: {n_body_ok}/{len(definitions)}")

    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract training data from Agda dump")
    parser.add_argument("--input", required=True, help="Path to JSON dump from Agda reflection")
    parser.add_argument("--output", default="cubical_training_tokens.json", help="Output path")
    args = parser.parse_args()

    print(f"Extracting from: {args.input}")
    samples = extract_training_data(args.input)
    print(f"Training samples: {len(samples)}")

    with open(args.output, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Saved to: {args.output}")

    for s in samples[:5]:
        print(f"\n  {s['name']}")
        print(f"    Type: {s['type']}")
        print(f"    Term: {s['term']}")
        print(f"    Tokens: {len(s['input_tokens'])} in, {len(s['target_tokens'])} out")


if __name__ == "__main__":
    main()
