"""
Agda Bridge — uses Agda as a live tokenizer and type checker.

Agda's frontend IS the tokenizer:
  Human-readable Agda → Agda parser/elaborator → Internal IR → our tokens

Agda's type checker IS the gate:
  Internal IR term → Agda type checks → valid/invalid

Uses Agda's --interaction-json protocol for bidirectional communication.
"""
from __future__ import annotations
import subprocess
import json
import re
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .ir import (
    Term, Var, App, Lam, Pi, Sigma, Pair, Fst, Snd,
    Universe, Set, Prop, Id, Refl, J, Nat, Zero, Suc, NatElim,
)
from .tokenizer import term_to_tokens, type_to_tokens, tokens_to_str


class AgdaBridge:
    """Bridge to Agda: uses Agda as tokenizer (frontend) and gate (type checker)."""

    def __init__(self, include_paths: Optional[List[str]] = None, timeout: int = 30):
        self.include_paths = include_paths or []
        self.timeout = timeout
        self.extract_dir = Path("extract")
        self._counter = 0

    def _agda_cmd(self, file_path: str) -> List[str]:
        cmd = ["agda"]
        for p in self.include_paths:
            cmd.append(f"--include-path={p}")
        cmd.append(f"--include-path={self.extract_dir}")
        cmd.append(file_path)
        return cmd

    def _run_agda(self, source: str) -> Tuple[bool, str, str]:
        """Write source to temp file, run Agda, return (success, stdout, stderr)."""
        self._counter += 1
        # Use a stable module name
        mod_name = f"AgdaBridge{self._counter}"
        file_path = self.extract_dir / f"{mod_name}.agda"
        source = source.replace("module BRIDGE", f"module {mod_name}")

        file_path.write_text(source, encoding='utf-8')
        try:
            result = subprocess.run(
                self._agda_cmd(str(file_path)),
                capture_output=True, text=True, timeout=self.timeout,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "timeout"
        finally:
            file_path.unlink(missing_ok=True)
            agdai = file_path.with_suffix('.agdai')
            agdai.unlink(missing_ok=True)

    def tokenize(self, agda_expr: str, context: str = "") -> Optional[Dict]:
        """Tokenize an Agda expression: parse → elaborate → extract Internal IR.

        Uses reflection to get the internal term representation.

        Args:
            agda_expr: Agda expression (e.g., "λ x → x")
            context: Additional Agda imports/definitions needed

        Returns:
            Dict with 'term' (our IR), 'tokens' (token IDs), 'type' (inferred type)
            or None if parsing/elaboration fails.
        """
        source = f"""{{-# OPTIONS --guardedness #-}}
module BRIDGE where

open import Agda.Builtin.Reflection renaming (returnTC to return)
open import Agda.Builtin.List
open import Agda.Builtin.Nat
open import Agda.Builtin.Equality
open import Agda.Builtin.Unit renaming (⊤ to Unit; tt to unit)
open import Extractor
open import DumpModule

{context}

target : _
target = {agda_expr}

dump-it = dump target
"""
        ok, stdout, stderr = self._run_agda(source)

        # Extract JSON from error output
        json_match = re.search(r'\{"name".*?\}(?=\s*when)', stderr, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                term = self._json_to_ir(data.get("type"))
                body = self._json_to_ir_body(data.get("body"))
                if body:
                    tokens = term_to_tokens(body)
                    return {
                        "term": body,
                        "type_ir": term,
                        "tokens": tokens,
                        "tokens_str": tokens_to_str(tokens),
                        "raw_json": data,
                    }
            except (json.JSONDecodeError, Exception):
                pass
        return None

    def type_check(self, agda_code: str) -> bool:
        """Use Agda as the gate: does this code type-check?

        Args:
            agda_code: Complete Agda module source

        Returns:
            True if Agda accepts it (type-checks), False otherwise.
        """
        ok, _, _ = self._run_agda(agda_code)
        return ok

    def check_term(self, term_str: str, type_str: str, context: str = "") -> bool:
        """Check if a term has the given type in Agda.

        Args:
            term_str: Agda expression for the term
            type_str: Agda expression for the expected type
            context: Additional imports/definitions

        Returns:
            True if term : type in Agda.
        """
        source = f"""{{-# OPTIONS --safe #-}}
module BRIDGE where

{context}

check : {type_str}
check = {term_str}
"""
        return self.type_check(source)

    def dump_definition(self, name: str, module_imports: str = "") -> Optional[Dict]:
        """Extract the internal representation of a named definition.

        Args:
            name: Qualified or unqualified Agda name
            module_imports: Agda import statements

        Returns:
            Dict with 'name', 'type', 'body' as JSON, or None.
        """
        source = f"""{{-# OPTIONS --guardedness #-}}
module BRIDGE where

open import Agda.Builtin.Reflection renaming (returnTC to return)
open import Agda.Builtin.List
open import Agda.Builtin.Unit renaming (⊤ to Unit; tt to unit)
open import Extractor
open import DumpModule

{module_imports}

dump-it = dump {name}
"""
        ok, stdout, stderr = self._run_agda(source)
        json_match = re.search(r'\{"name".*\}', stderr, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    # ── JSON → IR conversion ──────────────────────────────────────────

    def _json_to_ir(self, data) -> Optional[Term]:
        """Convert Agda reflected Term JSON to our dependent IR."""
        if data is None or not isinstance(data, dict):
            return None

        tag = data.get("tag", "")

        if tag == "var":
            idx = data.get("index", 0)
            if isinstance(idx, str):
                idx = int(idx)
            term = Var(idx)
            for arg in data.get("args", []):
                a = self._json_to_ir(arg.get("term", arg))
                if a is not None:
                    term = App(term, a)
            return term

        elif tag == "lam":
            body_data = data.get("body", {})
            body = self._json_to_ir(body_data.get("body", body_data))
            return Lam(body) if body else None

        elif tag == "pi":
            domain_data = data.get("domain", {})
            codomain_data = data.get("codomain", {})
            domain = self._json_to_ir(domain_data.get("term", domain_data))
            codomain = self._json_to_ir(codomain_data.get("body", codomain_data))
            if domain and codomain:
                return Pi(domain, codomain)
            return None

        elif tag == "sort":
            sort_data = data.get("sort", {})
            kind = sort_data.get("kind", "set")
            level = sort_data.get("level", 0)
            if isinstance(level, str):
                level = int(level) if level.isdigit() else 0
            if isinstance(level, dict):
                level = 0  # complex level expression, default to 0
            if "set" in kind:
                return Universe(Set(level))
            elif "prop" in kind:
                return Universe(Prop(level))
            return Universe(Set(0))

        elif tag == "def":
            name = data.get("name", "")
            args = [self._json_to_ir(a.get("term", a)) for a in data.get("args", [])]
            args = [a for a in args if a is not None]

            # Map known Agda definitions to our IR
            if "Nat" in name or "ℕ" in name:
                return Nat()
            elif "Sigma" in name or "Σ" in name:
                if len(args) >= 2: return Sigma(args[0], args[1])
            elif "≡" in name or "Id" in name or "＝" in name:
                if len(args) >= 3: return Id(args[0], args[1], args[2])

            # Build application chain for unknown defs
            term = Var(0)  # placeholder for named reference
            for a in args:
                term = App(term, a)
            return term

        elif tag == "con":
            name = data.get("name", "")
            args = [self._json_to_ir(a.get("term", a)) for a in data.get("args", [])]
            args = [a for a in args if a is not None]

            if "refl" in name:
                return Refl()
            elif "zero" in name:
                return Zero()
            elif "suc" in name and args:
                return Suc(args[0])

            # Generic pair
            if len(args) >= 2:
                return Pair(args[0], args[1])
            elif len(args) == 1:
                return args[0]
            return None

        elif tag == "lit":
            return Zero()  # simplified

        elif tag == "unknown":
            return None

        return None

    def _json_to_ir_body(self, data) -> Optional[Term]:
        """Convert body (list of clauses) to IR."""
        if isinstance(data, list) and data:
            # Take first clause's body
            clause = data[0]
            if isinstance(clause, dict):
                body = clause.get("body")
                if body:
                    return self._json_to_ir(body)
        elif isinstance(data, dict):
            return self._json_to_ir(data)
        return None
