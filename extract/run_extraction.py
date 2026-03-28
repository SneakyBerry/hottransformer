"""
Automated extraction of Agda internal terms from cubical-mini.

Usage:
    python extract/run_extraction.py --lib cubical-mini/src --output cubical_training_data.json
"""
import subprocess
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# cubical-mini flags from cubical-mini.agda-lib
CUBICAL_FLAGS = [
    "--erased-cubical",
    "--erasure",
    "--erased-matches",
    "--exact-split",
    "--guardedness",
    "--hidden-argument-puns",
    "--no-import-sorts",
    "--postfix-projections",
    "--qualified-instances",
]


def find_agda_files(lib_path: str) -> List[Path]:
    return sorted(Path(lib_path).rglob("*.agda"))


def extract_definitions(agda_file: Path, lib_root: Path) -> List[str]:
    """Extract top-level definition names from an Agda file."""
    names = []
    try:
        text = agda_file.read_text(encoding='utf-8')
    except Exception:
        return names

    in_private = False
    brace_depth = 0

    for line in text.split('\n'):
        stripped = line.strip()

        # Skip empty, comments, pragmas
        if not stripped or stripped.startswith('--') or stripped.startswith('{-#'):
            continue

        # Skip various declarations
        if stripped.startswith(('open', 'import', 'module', 'variable',
                               'infix', 'syntax', 'pattern', 'abstract',
                               'instance', 'private', 'unquoteDecl',
                               'unquoteDef', 'mutual', 'interleaved',
                               'opaque', 'unfolding')):
            continue

        # Skip data/record type declarations themselves
        if stripped.startswith('data ') or stripped.startswith('record '):
            continue

        # Match: unicode-aware identifier followed by colon (type signature)
        match = re.match(r'^([\w\-!\?\+\*\/\\\<\>\=\|\&\~\^\@\#\$\%₀-₉ᵃ-ᶻ\'\"\.]+)\s+:', stripped)
        if match:
            name = match.group(1)
            # Filter out noise
            if name not in ('where', 'data', 'record', 'field', 'constructor',
                            'let', 'in', 'do', 'with', 'rewrite'):
                names.append(name)

    return names


def module_name_from_path(agda_file: Path, lib_root: Path) -> str:
    rel = agda_file.relative_to(lib_root)
    return str(rel.with_suffix('')).replace('/', '.').replace('\\', '.')


def run_agda_dump(module_name: str, def_names: List[str],
                  lib_root: Path, extract_dir: Path) -> List[Dict]:
    """Run Agda to dump definitions from a module."""
    results = []

    for name in def_names:
        dump_file = extract_dir / "TempDump.agda"

        # Use the cubical-mini flags + our extractor
        # Use cubical flags for cubical-mini, plain for stdlib
        if 'agda-stdlib' in str(lib_root):
            opts = "--guardedness --erased-cubical"
        else:
            opts = "--erased-cubical --erasure --guardedness --no-import-sorts --erased-matches --hidden-argument-puns --postfix-projections --qualified-instances"

        dump_content = f"""{{-# OPTIONS {opts} #-}}
module TempDump where

open import {module_name}
open import Extractor
open import DumpModule

test = dump {name}
"""
        dump_file.write_text(dump_content, encoding='utf-8')

        try:
            # Auto-detect library from module path
            lib_flag = '--library=cubical-mini-0.6'
            if 'agda-stdlib' in str(lib_root):
                lib_flag = '--library=standard-library-2.4'

            cmd = ['agda',
                   lib_flag,
                   '--include-path=' + str(extract_dir),
                   str(dump_file)]

            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=60,
            )

            stderr = result.stdout  # combined output

            # Find JSON object in error output — it starts with {"name":
            json_match = re.search(r'(\{"name":.*\})\s*\n', stderr)
            if json_match:
                raw = json_match.group(1)
                try:
                    data = json.loads(raw)
                    results.append(data)
                except json.JSONDecodeError:
                    # JSON might have issues with 6+ notation, try fixing
                    fixed = re.sub(r'(\d)\+', r'\1', raw)
                    try:
                        data = json.loads(fixed)
                        results.append(data)
                    except json.JSONDecodeError:
                        pass
            elif '{"name":' in stderr:
                # Match found but regex failed — try multiline
                idx = stderr.index('{"name":')
                # Find matching closing brace
                depth = 0
                end = idx
                for ci, c in enumerate(stderr[idx:]):
                    if c == '{': depth += 1
                    elif c == '}': depth -= 1
                    if depth == 0:
                        end = idx + ci + 1
                        break
                raw = stderr[idx:end]
                fixed = re.sub(r'(\d)\+', r'\1', raw)
                try:
                    data = json.loads(fixed)
                    results.append(data)
                except json.JSONDecodeError:
                    pass

        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        # Delete only TempDump.agdai — force re-check of our dump file
        # but keep library .agdai cache intact for speed
        agdai = dump_file.with_suffix('.agdai')
        agdai.unlink(missing_ok=True)

    dump_file = extract_dir / "TempDump.agda"
    dump_file.unlink(missing_ok=True)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Agda terms from cubical-mini")
    parser.add_argument("--lib", default="cubical-mini/src", help="Path to cubical-mini/src")
    parser.add_argument("--output", default="cubical_training_data.json", help="Output JSON")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to process")
    parser.add_argument("--max-defs", type=int, default=10, help="Max definitions per file")
    parser.add_argument("--test", action="store_true", help="Test on first 5 files only")
    args = parser.parse_args()

    lib_root = Path(args.lib)
    extract_dir = Path("extract")

    if not lib_root.exists():
        print(f"Library not found: {lib_root}")
        sys.exit(1)

    agda_files = find_agda_files(lib_root)

    # Prioritize Foundations — they work and are most important for HoTT
    foundations = [f for f in agda_files if 'Foundations' in str(f)]
    others = [f for f in agda_files if 'Foundations' not in str(f)]
    agda_files = foundations + others

    if args.test:
        agda_files = agda_files[:10]
    elif args.max_files:
        agda_files = agda_files[:args.max_files]

    print(f"  ({len(foundations)} Foundations, {len(others)} other)")

    print(f"Found {len(agda_files)} Agda files in {lib_root}")

    # First, test that our extractor works with cubical flags
    print("Testing Agda setup...")
    test_ok = subprocess.run(
        ['agda', '--erased-cubical', '--include-path=' + str(lib_root),
         '--include-path=' + str(extract_dir),
         str(extract_dir / "Extractor.agda")],
        capture_output=True, text=True, timeout=60,
    )
    if test_ok.returncode != 0:
        print(f"Extractor.agda failed to compile with cubical flags:")
        print(test_ok.stderr[:500])
        # Try to fix
        print("Attempting without --erased-cubical for extractor...")

    all_results = []
    total_defs_tried = 0

    for agda_file in tqdm(agda_files, desc="Extracting"):
        module_name = module_name_from_path(agda_file, lib_root)
        def_names = extract_definitions(agda_file, lib_root)

        if not def_names:
            continue

        def_names = def_names[:args.max_defs]
        total_defs_tried += len(def_names)

        results = run_agda_dump(module_name, def_names, lib_root, extract_dir)
        all_results.extend(results)

        # Progress
        if all_results and len(all_results) % 10 == 0:
            tqdm.write(f"  ... {len(all_results)} definitions extracted so far")

    print(f"\nTried {total_defs_tried} definitions across {len(agda_files)} files")
    print(f"Extracted {len(all_results)} definitions")

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {args.output}")

    # Show examples
    for d in all_results[:5]:
        name = d.get("name", "?")
        print(f"  {name}")


if __name__ == "__main__":
    main()
