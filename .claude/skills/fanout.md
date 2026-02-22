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
# Explicit models
uv run fanout run "YOUR PROMPT" \
  -m openai/gpt-4o-mini -m anthropic/claude-haiku-4 \
  -e latency -e cost \
  -s top-k -r 3 -n 6

# Model set (weighted random sampling)
uv run fanout run "YOUR PROMPT" \
  -M coding -e script --eval-script ./eval.sh \
  -s alphaevolve -r 5 -n 8

# Verbose output with syntax-highlighted previews
uv run fanout run "YOUR PROMPT" -m openai/gpt-4o-mini -n 5 -r 3 -v
```

### Sample only
```bash
uv run fanout sample "YOUR PROMPT" -m openai/gpt-4o-mini -n 3
uv run fanout sample "YOUR PROMPT" -M diverse -n 5 -v
```

### Evaluate existing solutions
```bash
uv run fanout evaluate RUN_ID -e latency -e accuracy --reference "expected answer"
uv run fanout evaluate RUN_ID -e script --eval-script ./eval.sh
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

### List available components
```bash
uv run fanout list-evaluators
uv run fanout list-strategies
uv run fanout list-materializers
uv run fanout list-model-sets
```

## Options

### Model selection (mutually exclusive)
- `-m/--model`: Model to sample (repeatable). Default: `openai/gpt-4o-mini`
- `-M/--model-set`: Named model set for weighted random sampling. Available: `coding`, `diverse`, `large`, `math-proving`, `small`

### Sampling
- `-n/--n-samples`: Total samples per round. Distributed round-robin across models when using `-m`. Default: `5`
- `--temperature`: Sampling temperature. Default: `0.7`
- `--max-tokens`: Max tokens per response. Default: `2048`

### Evaluation
- `-e/--evaluator`: Evaluator to apply (repeatable). Default: `latency`, `cost`
  - `latency` — lower latency scores higher
  - `cost` — fewer tokens scores higher
  - `accuracy` — similarity to `--reference` answer
  - `script` — runs `--eval-script` against materialized output
- `--eval-script`: Path to eval script (auto-adds `-e script`)
- `--materializer`: How to present solutions to eval scripts. Default: `file`
  - `file` — writes to a temp file (use with `--file-ext`)
  - `stdin` — pipes via stdin
  - `worktree` — applies as a unified diff in a git worktree
- `--file-ext`: File extension for file materializer. Default: `.py`
- `--reference`: Reference answer for the accuracy evaluator

### Strategy & selection
- `-s/--strategy`: Selection strategy. Default: `top-k`
  - `top-k` — select K highest-scoring solutions
  - `weighted` — select with probability proportional to score
  - `rsa` — Recursive Self-Aggregation: feed K parent solutions back into each prompt
  - `alphaevolve` — score-aware selection + annotated aggregation + biased subsampling
  - `island` — evolve subpopulations per model with periodic migration
  - `map-elites` — best solution per behavioral dimension cell
- `-r/--rounds`: Number of evolutionary rounds. Default: `1`
- `--k`: Selection size. Default: `3`
- `--k-agg`: Number of parent solutions per aggregation prompt (RSA/alphaevolve). Default: `3`

### Output
- `-v/--verbose`: Show per-solution details with syntax-highlighted code previews

## Environment

- `OPENROUTER_API_KEY`: Required. Your OpenRouter API key.

## Data

Results are stored in `.fanout/fanout.db` (SQLite) in the current project directory.
