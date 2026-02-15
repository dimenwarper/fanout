#!/usr/bin/env bash
set -euo pipefail

# install.sh — Install fanout CLI system-wide and register Claude Code skill
#
# Usage:
#   curl -fsSL <raw-url>/install.sh | bash
#   # or from a local clone:
#   ./install.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAUDE_COMMANDS_DIR="${HOME}/.claude/commands"

echo "==> Installing fanout CLI system-wide with uv..."
uv tool install --force "${REPO_DIR}"
echo "    Done. $(fanout --help | head -1)"

echo ""
echo "==> Installing Claude Code /fanout skill..."
mkdir -p "${CLAUDE_COMMANDS_DIR}"
cp "${REPO_DIR}/.claude/skills/setup.md" "${CLAUDE_COMMANDS_DIR}/fanout-setup.md"

# Create the /fanout slash command for Claude Code
cat > "${CLAUDE_COMMANDS_DIR}/fanout.md" <<'SKILL'
---
name: fanout
description: Run fanout — fan out LLM samples, evaluate, and select the best
user_invocable: true
---

# /fanout — Run a fanout evolutionary loop

Run the `fanout` CLI to sample multiple LLMs, evaluate outputs, and select the best solutions.

## Instructions

1. Ask the user for:
   - The **prompt** to fan out
   - (Optional) which **models** or **model set** to use (default: coding set)
   - (Optional) number of **samples** (default: 5)
   - (Optional) number of **rounds** (default: 1)
   - (Optional) **evaluators** (default: latency, cost)
   - (Optional) an **eval script** path for script-based evaluation

2. Build and run the appropriate `fanout run` command. Examples:

   ```bash
   # Basic run with model set
   fanout run "Write a Python function that reverses a linked list" -M coding -N 5 -e latency -e cost

   # With script evaluation
   fanout run "Write a sort function" -M coding -N 5 --eval-script /path/to/test.sh --materializer file

   # Multi-round with explicit models
   fanout run "Write a haiku" -m openai/gpt-4o-mini -m anthropic/claude-3-haiku -r 3 -n 2 -e latency -s top-k
   ```

3. Show the results table to the user and highlight the winning solution(s).

## Available subcommands

- `fanout run` — full evolutionary loop
- `fanout sample` — sample only
- `fanout evaluate` — evaluate only
- `fanout select` — select only
- `fanout store` — inspect runs
- `fanout list-evaluators` / `fanout list-materializers` / `fanout list-strategies` / `fanout list-model-sets`
SKILL

echo "    Installed: ${CLAUDE_COMMANDS_DIR}/fanout.md"
echo "    Installed: ${CLAUDE_COMMANDS_DIR}/fanout-setup.md"

echo ""
echo "==> All done!"
echo ""
echo "  fanout CLI:     $(which fanout)"
echo "  Claude skills:  /fanout, /fanout-setup"
echo ""
echo "  Quick test:     fanout list-evaluators"
echo "  Claude Code:    Type /fanout in any Claude Code session"
