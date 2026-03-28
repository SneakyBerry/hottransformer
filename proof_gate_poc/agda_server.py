"""
Fast Agda gate — reuses cached interface files, minimal overhead.

Instead of interaction protocol (complex), we:
1. Keep a warm work directory with pre-compiled imports cached as .agdai
2. Write check file → agda → read result → keep .agdai cache
3. First call is slow (compiles imports), subsequent calls fast (cached)

For production: extract Agda's type checker into standalone Haskell library.
Agda is 200K lines, type checker core is ~5K. Only need:
  - Agda.TypeChecking.Rules.Term (check/infer)
  - Agda.TypeChecking.Reduce (normalization)
  - Agda.Syntax.Internal (IR)

This is a task for the Haskell-literate friend.
"""
from __future__ import annotations
import subprocess
import os
import time
import re
import threading
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class CheckResult:
    success: bool
    errors: str = ""
    elapsed_ms: float = 0


class AgdaServer:
    """Fast Agda type checker with warm cache."""

    def __init__(self, libraries: Optional[List[str]] = None, timeout: int = 30):
        self.timeout = timeout
        self.libraries = libraries or []
        self._lock = threading.Lock()
        self._counter = 0
        self._work_dir = Path("proof_gate_poc/checkpoints/agda_work")
        self._work_dir.mkdir(parents=True, exist_ok=True)
        # Pre-warm: compile a trivial file to cache builtins
        self._warm()

    def _warm(self):
        """Pre-compile builtins so subsequent checks are fast."""
        warm_file = self._work_dir / "Warm.agda"
        warm_file.write_text("module Warm where\nopen import Agda.Builtin.Nat\nopen import Agda.Builtin.Equality\n")
        subprocess.run(
            self._agda_cmd(str(warm_file)),
            capture_output=True, text=True, timeout=self.timeout,
        )
        # Keep .agdai — cache is warm now

    def _agda_cmd(self, file_path: str) -> List[str]:
        cmd = ["agda", f"--include-path={self._work_dir}"]
        for lib in self.libraries:
            cmd.append(f"--library={lib}")
        cmd.append(file_path)
        return cmd

    def check(self, code: str) -> CheckResult:
        """Type-check Agda code."""
        with self._lock:
            t0 = time.time()
            self._counter += 1
            mod_name = f"Check{self._counter}"
            file_path = self._work_dir / f"{mod_name}.agda"

            # Fix module name
            code_fixed = re.sub(r'module\s+\S+', f'module {mod_name}', code, count=1)
            if 'module ' not in code_fixed:
                code_fixed = f"module {mod_name} where\n{code_fixed}"

            file_path.write_text(code_fixed, encoding='utf-8')

            try:
                result = subprocess.run(
                    self._agda_cmd(str(file_path)),
                    capture_output=True, text=True, timeout=self.timeout,
                )
                elapsed = (time.time() - t0) * 1000
                success = result.returncode == 0
                errors = result.stderr.strip() if not success else ""
                # Clean paths from errors
                errors = errors.replace(str(file_path), f"{mod_name}.agda")

                return CheckResult(success=success, errors=errors, elapsed_ms=elapsed)

            except subprocess.TimeoutExpired:
                return CheckResult(success=False, errors="timeout", elapsed_ms=self.timeout * 1000)
            finally:
                file_path.unlink(missing_ok=True)
                # Keep .agdai for cache!

    def check_term(self, term: str, type_str: str, context: str = "") -> bool:
        """Check if term : type."""
        code = f"""module Check where
{context}

check : {type_str}
check = {term}
"""
        return self.check(code).success

    def shutdown(self):
        """Cleanup work dir."""
        import shutil
        if self._work_dir.exists():
            for f in self._work_dir.glob("Check*.agda*"):
                f.unlink(missing_ok=True)


_server = None

def get_agda_server(libraries=None) -> AgdaServer:
    global _server
    if _server is None:
        _server = AgdaServer(libraries=libraries)
    return _server
