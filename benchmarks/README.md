# Benchmarks

Curated benchmarks for evaluating fanout strategies (especially RSA) against public baselines.

## Overview

| Benchmark | Domain | Scoring | Runtime Needs | Tasks |
|-----------|--------|---------|---------------|-------|
| [kernelbench/](kernelbench/) | CUDA kernel optimization | Correctness + speedup (0.0-1.0) | CUDA GPU + PyTorch | 5 |
| [codeevolve/](codeevolve/) | Algorithm discovery | benchmark_ratio vs AlphaEvolve (0.0-1.0) | Python + NumPy | 4 |
| [minif2f/](minif2f/) | Formal theorem proving | Binary pass/fail (0.0 or 1.0) | Lean 4 + Mathlib | 5 |

## Running benchmarks

Each benchmark has a `run_benchmark.py` that automates the full loop: iterates tasks, constructs prompts, runs fanout's evolutionary loop, and prints a comparison table.

```bash
# Install benchmark deps
uv sync --extra benchmarks

# CodeEvolve — no GPU or Lean needed, easiest to start with
uv run --extra benchmarks python benchmarks/codeevolve/run_benchmark.py \
  -m openai/gpt-4o-mini -r 3 -n 2

# Compare strategies head-to-head
uv run --extra benchmarks python benchmarks/codeevolve/run_benchmark.py \
  -s top-k rsa -r 5 --k-agg 2

# Run a single task
uv run --extra benchmarks python benchmarks/codeevolve/run_benchmark.py \
  --tasks circle_packing -s rsa -r 5

# KernelBench (requires CUDA GPU)
uv run --extra benchmarks python benchmarks/kernelbench/run_benchmark.py \
  -m openai/gpt-4o -r 3

# miniF2F (requires Lean 4 / elan)
uv run --extra benchmarks python benchmarks/minif2f/run_benchmark.py \
  -m anthropic/claude-sonnet-4 -r 5
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | all | Which tasks to run |
| `-s` / `--strategy` | `top-k rsa` | Strategies to compare (runs each task with each strategy) |
| `-m` / `--model` | `gpt-4o-mini` | Model(s) to use (repeatable) |
| `-r` / `--rounds` | 3-5 | Evolutionary rounds |
| `-n` | 2-3 | Samples per model per round |
| `-k` | 3 | Selection size |
| `--k-agg` | 2-3 | RSA aggregation size |

## Low-level eval scripts

Each benchmark also has an `eval.sh` for scoring individual solutions, compatible with fanout's `--eval-script` flag:

```bash
# Score a single solution
benchmarks/codeevolve/eval.sh solution.py circle_packing
benchmarks/kernelbench/eval.sh solution.py tasks/matmul.py
benchmarks/minif2f/eval.sh proof.lean
```

## Structure

```
benchmarks/
├── README.md
├── kernelbench/
│   ├── README.md
│   ├── eval.sh              # Correctness + speedup scorer
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 5 Level-1 PyTorch reference ops
├── codeevolve/
│   ├── README.md
│   ├── eval.sh              # Constraint validation + benchmark ratio
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 4 optimization problems with baselines
└── minif2f/
    ├── README.md
    ├── eval.sh              # Lean 4 typecheck (binary)
    ├── run_benchmark.py     # Full benchmark runner
    └── tasks/               # 5 olympiad theorem statements
```

## Adding new benchmarks

A benchmark needs:
1. **Task files** in `tasks/` — the problem definition (prompt source material)
2. **`eval.sh`** — takes a solution file path as `$1`, prints a score (0.0-1.0) on the last line of stdout
3. **`run_benchmark.py`** — iterates tasks, builds prompts, calls fanout internals, prints results
4. **`README.md`** — how to run it, what's needed

The eval script interface matches fanout's `--eval-script` contract, so any benchmark here works directly with `fanout run`.
