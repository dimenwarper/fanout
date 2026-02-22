#!/usr/bin/env python3
"""miniF2F eval script for fanout.

Usage: ./eval.py <solution_file> [task_file]

The solution file should contain a complete Lean 4 file with the theorem
and its proof (no `sorry`). This script:
  1. Creates a temporary Lean 4 project with mathlib
  2. Copies the solution in
  3. Runs `lake build` to typecheck
  4. Prints 1.0 (QED) or 0.0 (failed) on the last line

Requires: elan (Lean 4 version manager), lake
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```lean ... ```) from LLM output."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <solution_file>", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    solution_path = Path(sys.argv[1])

    # Strip code fences and check for sorry
    content = strip_code_fences(solution_path.read_text())
    if "sorry" in content:
        print("Proof contains sorry", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    # Check lean/lake are available
    if shutil.which("lake") is None:
        print("lake not found. Install elan: https://github.com/leanprover/elan", file=sys.stderr)
        print("0.0")
        sys.exit(0)

    # Create temp project, typecheck, clean up
    tmpdir = Path(tempfile.mkdtemp())
    try:
        # Initialize a Lean 4 project with mathlib
        subprocess.run(
            ["lake", "init", "check", "math"],
            cwd=tmpdir, capture_output=True, check=True,
        )
        project_dir = tmpdir / "check"

        # Write cleaned solution as Main.lean
        (project_dir / "Main.lean").write_text(content)

        # Try to build (typecheck)
        print("Typechecking proof...", file=sys.stderr)
        result = subprocess.run(
            ["lake", "build"],
            cwd=project_dir, capture_output=True, text=True,
        )

        output = result.stdout + result.stderr
        print(output, file=sys.stderr)

        if result.returncode != 0 or "error:" in output:
            print("Proof failed to typecheck", file=sys.stderr)
            print("0.0")
        else:
            print("Proof typechecked successfully (QED)", file=sys.stderr)
            print("1.0")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("0.0")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
