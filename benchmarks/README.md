# Benchmarks

Curated benchmarks for evaluating fanout strategies (especially RSA) against public baselines.

## Overview

| Benchmark | Domain | Scoring | Runtime Needs | Tasks |
|-----------|--------|---------|---------------|-------|
| [kernelbench/](kernelbench/) | CUDA kernel optimization | Correctness + speedup (0.0-1.0) | CUDA GPU + PyTorch | 5 |
| [codeevolve/](codeevolve/) | Algorithm discovery | benchmark_ratio vs AlphaEvolve (0.0-1.0) | Python + NumPy | 4 |
| [minif2f/](minif2f/) | Formal theorem proving | Binary pass/fail (0.0 or 1.0) | Lean 4 + Mathlib | 5 |
| [molopt/](molopt/) | Molecular optimization | Min property score, diversity-enforced (0.0-1.0) | Python + RDKit | 4 |
| [mnist-weights/](mnist-weights/) | Raw neural network weights | Test accuracy (0.0-1.0) | Python + scikit-learn | 4 |
| [cifar10-weightgen/](cifar10-weightgen/) | CNN weight generation | Test accuracy (0.0-1.0) | Python + PyTorch | 3 |

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

# MolOpt — evolve SMILES strings for drug-likeness
uv run --extra benchmarks python benchmarks/molopt/run_benchmark.py \
  --tasks maximize_qed -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# MNIST-Weights — write raw MLP weights for digit classification
uv run --extra benchmarks python benchmarks/mnist-weights/run_benchmark.py \
  --tasks binary_0v1 -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# CIFAR-10 WeightGen — generate CNN weights using mathematical patterns
uv run --extra benchmarks python benchmarks/cifar10-weightgen/run_benchmark.py \
  --tasks generate_cnn_small -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

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

Each benchmark also has an `eval.py` for scoring individual solutions, compatible with fanout's `--eval-script` flag:

```bash
# Score a single solution
benchmarks/codeevolve/eval.py solution.py circle_packing
benchmarks/molopt/eval.py solution.py maximize_qed
benchmarks/mnist-weights/eval.py solution.py binary_0v1
benchmarks/cifar10-weightgen/eval.py solution.py generate_cnn_small
benchmarks/kernelbench/eval.py solution.py tasks/matmul.py
benchmarks/minif2f/eval.py proof.lean
```

## Structure

```
benchmarks/
├── README.md
├── kernelbench/
│   ├── README.md
│   ├── eval.py              # Correctness + speedup scorer
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 5 Level-1 PyTorch reference ops
├── codeevolve/
│   ├── README.md
│   ├── eval.py              # Constraint validation + benchmark ratio
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 4 optimization problems with baselines
├── minif2f/
│   ├── README.md
│   ├── eval.py              # Lean 4 typecheck (binary)
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 5 olympiad theorem statements
├── molopt/
│   ├── README.md
│   ├── eval.py              # SMILES parsing + property scoring
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 4 molecular optimization tasks
├── mnist-weights/
│   ├── README.md
│   ├── eval.py              # Numpy forward pass + accuracy
│   ├── run_benchmark.py     # Full benchmark runner
│   └── tasks/               # 4 digit classification tasks
└── cifar10-weightgen/
    ├── README.md
    ├── eval.py              # PyTorch inference + accuracy
    ├── run_benchmark.py     # Full benchmark runner
    └── tasks/               # 3 CNN weight generation tasks
```

## Adding new benchmarks

A benchmark needs:
1. **Task files** in `tasks/` — the problem definition (prompt source material)
2. **`eval.py`** — takes a solution file path as `$1`, prints a score (0.0-1.0) on the last line of stdout
3. **`run_benchmark.py`** — iterates tasks, builds prompts, calls fanout internals, prints results
4. **`README.md`** — how to run it, what's needed

The eval script interface matches fanout's `--eval-script` contract, so any benchmark here works directly with `fanout run`.
