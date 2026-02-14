---
name: fanout
description: Sample multiple LLMs, evaluate outputs, and select the best solutions using evolutionary strategies
user_invocable: true
---

# /fanout — LLM Sampling & Selection

Run the fanout CLI to sample multiple LLMs, evaluate their outputs, and iteratively select the best solutions.

## Usage

Dispatch to the appropriate subcommand based on user intent:

### Full evolutionary run
```bash
uv run fanout run "YOUR PROMPT" \
  -m openai/gpt-4o-mini -m anthropic/claude-3-haiku \
  -e latency -e cost \
  -s top-k -r 3 -n 2
```

### Sample only
```bash
uv run fanout sample "YOUR PROMPT" -m openai/gpt-4o-mini -n 3
```

### Evaluate existing solutions
```bash
uv run fanout evaluate RUN_ID -e latency -e accuracy --reference "expected answer"
```

### Select best solutions
```bash
uv run fanout select RUN_ID -s top-k --k 3
```

### Inspect stored runs
```bash
uv run fanout store          # list all runs
uv run fanout store RUN_ID   # inspect a specific run
```

### List available evaluators and strategies
```bash
uv run fanout list-evaluators
uv run fanout list-strategies
```

## Options

- `-m/--model`: Model to sample (repeatable). Default: `openai/gpt-4o-mini`
- `-e/--evaluator`: Evaluator to apply (repeatable). Default: `latency`, `cost`
- `-s/--strategy`: Selection strategy. Default: `top-k`
- `-r/--rounds`: Number of evolutionary rounds. Default: `1`
- `-n`: Samples per model per round. Default: `1`
- `--k`: Selection size for top-k/weighted. Default: `3`
- `--reference`: Reference answer for the accuracy evaluator
- `--temperature`: Sampling temperature. Default: `0.7`
- `--max-tokens`: Max tokens per response. Default: `2048`

## Environment

- `OPENROUTER_API_KEY`: Required. Your OpenRouter API key.

## Data

Results are stored in `.fanout/fanout.db` (SQLite) in the current project directory.
