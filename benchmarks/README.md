# Benchmarks

Curated benchmarks for evaluating fanout strategies (especially RSA) against public baselines.

## Overview

| Benchmark | Domain | Scoring | Runtime Needs | Tasks |
|-----------|--------|---------|---------------|-------|
| [kernelbench/](kernelbench/) | CUDA kernel optimization | Correctness + speedup (0.0-1.0) | CUDA GPU + PyTorch | 5 |
| [codeevolve/](codeevolve/) | Algorithm discovery | benchmark_ratio vs AlphaEvolve (0.0-1.0) | Python + NumPy | 4 |
| [minif2f/](minif2f/) | Formal theorem proving | Binary pass/fail (0.0 or 1.0) | Lean 4 + Mathlib | 5 |

## Quick start

Each benchmark has an `eval.sh` that scores a solution file and prints a float on the last line, compatible with fanout's `--eval-script` flag.

```bash
# Example: RSA on circle packing (no GPU required)
cd codeevolve
fanout run "$(cat tasks/circle_packing.py)

Improve circle_packing26() to maximize sum of radii for 26 non-overlapping circles in the unit square. Output only the Python file." \
  -m openai/gpt-4o-mini -n 3 \
  -s rsa --k-agg 2 -r 5 \
  --eval-script "./eval.sh" --materializer file --file-ext .py \
  -e script
```

## Structure

```
benchmarks/
├── README.md
├── kernelbench/          # CUDA kernel optimization (from KernelBench)
│   ├── README.md
│   ├── eval.sh           # Correctness + speedup scorer
│   └── tasks/            # 5 Level-1 PyTorch reference ops
├── codeevolve/           # Algorithm discovery (from CodeEvolve/AlphaEvolve)
│   ├── README.md
│   ├── eval.sh           # Constraint validation + benchmark ratio
│   └── tasks/            # 4 optimization problems with baselines
└── minif2f/              # Formal proofs (from miniF2F)
    ├── README.md
    ├── eval.sh           # Lean 4 typecheck (binary)
    └── tasks/            # 5 olympiad theorem statements
```

## Adding new benchmarks

A benchmark needs:
1. **Task files** in `tasks/` — the problem definition (prompt source material)
2. **`eval.sh`** — takes a solution file path as `$1`, prints a score (0.0-1.0) on the last line of stdout
3. **`README.md`** — how to run it, what's needed

The eval script interface matches fanout's `--eval-script` contract, so any benchmark here works directly with `fanout run`.
