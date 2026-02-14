# fanout

Sample multiple LLMs, evaluate outputs, and iteratively select the best solutions using evolutionary strategies.

## Install

```bash
uv sync
```

Requires `OPENROUTER_API_KEY` environment variable.

## Quick Start

```bash
# Full evolutionary run: 3 rounds, 2 models, 2 evaluators
uv run fanout run "Write a haiku about recursion" \
  -m openai/gpt-4o-mini -m anthropic/claude-3-haiku \
  -e latency -e cost \
  -s top-k -r 3 -n 2

# Or step by step:
uv run fanout sample "Write a haiku" -m openai/gpt-4o-mini -n 3
uv run fanout evaluate <RUN_ID> -e latency -e cost
uv run fanout select <RUN_ID> -s top-k --k 2
```

## Commands

| Command | Description |
|---------|-------------|
| `fanout run` | Full loop: sample → evaluate → select × N rounds |
| `fanout sample` | Fan out a prompt to models |
| `fanout evaluate` | Score solutions with evaluators |
| `fanout select` | Pick best solutions using a strategy |
| `fanout store` | List runs or inspect a specific run |
| `fanout list-evaluators` | Show available evaluators |
| `fanout list-strategies` | Show available strategies |

## Evaluators

| Name | Description |
|------|-------------|
| `latency` | Scores inversely proportional to response time |
| `accuracy` | Scores by similarity to a reference answer |
| `cost` | Scores inversely proportional to token cost |

## Selection Strategies

| Name | Description |
|------|-------------|
| `top-k` | Select the K highest-scoring solutions |
| `weighted` | Probability proportional to score |
| `map-elites` | Best solution per behavioral dimension cell |
| `island` | Subpopulation evolution with migration |

## Architecture

```
prompt → [model₁, model₂, ...] → solutions → [eval₁, eval₂, ...] → scores → strategy → selected
                                                                                    ↓
                                                                              next round (repeat)
```

Data is stored in `.fanout/fanout.db` (SQLite, WAL mode) in the project directory.
