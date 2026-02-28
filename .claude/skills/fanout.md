---
name: fanout
description: Sample multiple LLMs, evaluate outputs, and select the best solutions using evolutionary strategies
user_invocable: true
---

# /fanout — LLM Sampling & Selection

Run the fanout CLI to sample multiple LLMs, evaluate their outputs, and iteratively select the best solutions.

## Usage

Dispatch to the appropriate subcommand based on user intent:

### Full evolutionary run (sample mode)
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

# Darwinian strategy — sigmoid + novelty-bonus selection
uv run fanout run "YOUR PROMPT" \
  -M coding -e script --eval-script ./eval.sh \
  -s darwinian -r 5 -n 8

# With shared memory bank (learnings persist across rounds)
uv run fanout run "YOUR PROMPT" \
  -M coding -e script --eval-script ./eval.sh \
  -s darwinian -r 5 -n 8 --memory

# Pareto-front selection — preserves non-dominated trade-offs across evaluators
uv run fanout run "YOUR PROMPT" \
  -M coding -e latency -e accuracy --reference "expected" \
  -s pareto -r 4 -n 8

# Epsilon-greedy — 20% random exploration, 80% greedy
uv run fanout run "YOUR PROMPT" \
  -M coding -e script --eval-script ./eval.sh \
  -s epsilon-greedy --epsilon 0.2 -r 4 -n 8

# Reflective mutation — LLM diagnoses failures and injects improvement brief each round
uv run fanout run "YOUR PROMPT" \
  -M coding -e script --eval-script ./eval.sh \
  -s darwinian -r 5 -n 8 --reflection

# Verbose output with syntax-highlighted previews
uv run fanout run "YOUR PROMPT" -m openai/gpt-4o-mini -n 5 -r 3 -v

# Full untruncated solutions
uv run fanout run "YOUR PROMPT" -m openai/gpt-4o-mini -n 5 -r 3 --full
```

### Agent mode
```bash
# Launch concurrent agents that iteratively improve solutions
uv run fanout run "YOUR PROMPT" --mode agent \
  --n-agents 5 --max-steps 10 \
  --eval-script ./eval.sh -s top-k -k 3

# Agent mode with shared memory bank
uv run fanout run "YOUR PROMPT" --mode agent \
  -M coding --n-agents 5 --max-steps 10 --memory

# Atomic launch (without selection)
uv run fanout launch "YOUR PROMPT" -n 3 --max-steps 10 --eval-script ./eval.sh -v
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
uv run fanout select RUN_ID -s darwinian --k 3
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

### Workflow mode
- `--mode`: Workflow mode — `sample` (default) or `agent`
- `--n-agents`: Number of concurrent agents for agent mode. Default: `3`
- `--max-steps`: Max iterations per agent. Default: `10`

### Sampling
- `-n/--n-samples`: Total samples per round (sample mode). Default: `5`
- `--temperature`: Sampling temperature. Default: `0.7`
- `--max-tokens`: Max tokens per response. Default: `16384`

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
- `--eval-timeout`: Timeout per evaluation in seconds. Default: `60`
- `--reference`: Reference answer for the accuracy evaluator
- `-p/--eval-concurrency`: Max parallel evaluations. Default: `1`

### Strategy & selection
- `-s/--strategy`: Selection strategy. Default: `top-k`
  - `top-k` — select K highest-scoring solutions (pure elitism)
  - `weighted` — select with probability proportional to score
  - `rsa` — Recursive Self-Aggregation: feed K parent solutions back into each prompt
  - `alphaevolve` — score-aware selection + annotated aggregation + biased subsampling
  - `island` — evolve subpopulations per model with periodic migration
  - `map-elites` — best solution per behavioral dimension cell
  - `darwinian` — **sigmoid-scaled selection with novelty bonus** (see below)
  - `pareto` — **Pareto-front selection** across all evaluator objectives; preserves non-dominated trade-offs
  - `epsilon-greedy` — with probability `--epsilon` pick a random candidate, otherwise pick the best
- `-r/--rounds`: Number of evolutionary rounds. Default: `1`
- `--k`: Selection size. Default: `3`
- `--k-agg`: Number of parent solutions per aggregation prompt (RSA/alphaevolve). Default: `6`
- `--epsilon`: Exploration probability for epsilon-greedy strategy. Default: `0.1`

### Memory bank
- `--memory`: Enable shared memory bank across rounds/agents. When set:
  - **Sample mode**: after each round, learnings from the best and worst selected solutions are written to the memory bank and injected into the next round's prompt as context
  - **Agent mode**: each agent gets `write_memory` and `read_memories` tools and a memory-aware system prompt; agents share observations, hypotheses, and learnings in real time

### Reflective mutation
- `--reflection/--no-reflection`: After each round, call an LLM to diagnose failure modes from execution traces and prepend the resulting improvement brief to the next round's prompt (inspired by GEPA)
- `--reflection-model`: Model used for the reflection call. Default: `google/gemini-2.0-flash-001`

### Output
- `-v/--verbose`: Show per-solution details with syntax-highlighted code previews
- `--full`: Show full untruncated solutions
- `--record`: Record the run to a persistent summary file (benchmark runners)

## Darwinian strategy

The `darwinian` strategy avoids premature convergence using:

```
weight = sigmoid(sharpness × (score − midpoint)) × (1 / (1 + novelty_weight × times_used_as_parent))
```

- **Sigmoid midpoint** defaults to the 75th percentile of current scores (`p75`), so pressure auto-scales to the population
- **Novelty bonus** penalises solutions already used as parents many times, forcing exploration even when one solution leads
- Configurable via the Python API:

```python
from fanout.workflow import SampleWorkflow

wf = SampleWorkflow()
result = wf.run(
    prompt="...",
    strategy="darwinian",
    k=3,
    rounds=5,
    darwinian_sharpness=5.0,       # softer selection pressure (default 10.0)
    darwinian_midpoint="p50",      # midpoint = median score (default "p75")
    darwinian_novelty_weight=2.0,  # stronger novelty pressure (default 1.0)
)
```

## Environment

- `OPENROUTER_API_KEY`: Required. Your OpenRouter API key.

## Data

Results are stored in Redis (`localhost:6379`, key prefix `fanout:`). If Redis is unavailable, an ephemeral in-memory store is used. Use `fanout store` to list and inspect runs.
