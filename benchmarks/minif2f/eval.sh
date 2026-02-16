#!/bin/bash
# miniF2F eval script for fanout.
#
# Usage: ./eval.sh <solution_file> [task_file]
#
# The solution file should contain a complete Lean 4 file with the theorem
# and its proof (no `sorry`). This script:
#   1. Creates a temporary Lean 4 project with mathlib
#   2. Copies the solution in
#   3. Runs `lake build` to typecheck
#   4. Prints 1.0 (QED) or 0.0 (failed) on the last line
#
# Requires: elan (Lean 4 version manager), lake

set -euo pipefail

SOLUTION_FILE="$1"
TASK_FILE="${2:-}"

# Check for sorry — an incomplete proof scores 0
if grep -q "sorry" "$SOLUTION_FILE"; then
    echo "Proof contains sorry" >&2
    echo "0.0"
    exit 0
fi

# Check lean/lake are available
if ! command -v lake &>/dev/null; then
    echo "lake not found. Install elan: https://github.com/leanprover/elan" >&2
    echo "0.0"
    exit 0
fi

# Create temp project
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

cd "$TMPDIR"

# Initialize a Lean 4 project with mathlib
lake init check math
cd check

# Copy solution as Main.lean
cp "$SOLUTION_FILE" Main.lean

# Try to build (typecheck)
echo "Typechecking proof..." >&2
if lake build 2>&1 | tee /dev/stderr | grep -q "error:"; then
    echo "Proof failed to typecheck" >&2
    echo "0.0"
else
    echo "Proof typechecked successfully (QED)" >&2
    echo "1.0"
fi
