# CodeEvolve

Algorithm discovery and optimization tasks adapted from [CodeEvolve](https://github.com/inter-co/science-codeevolve), which reproduces the benchmarks used to evaluate Google DeepMind's AlphaEvolve.

## Tasks included

| Task | Function | Benchmark (AlphaEvolve) | Description |
|------|----------|------------------------|-------------|
| `circle_packing.py` | `circle_packing26()` | sum_radii = 2.636 | Pack 26 circles in unit square, maximize total radii |
| `kissing_number.py` | `kissing_number11()` | 593 points | Max integer points in 11D with norm constraint |
| `first_autocorr.py` | `first_autocorrelation()` | 1/C1 > 0.665 | Minimize autocorrelation constant C1 |
| `heilbronn_triangle.py` | `heilbronn_triangle11()` | min_area = 0.0365 | Place 11 points to maximize minimum triangle area |

## Eval

```bash
chmod +x eval.py

# Score a solution for a specific task
./eval.py solution.py circle_packing
./eval.py solution.py kissing_number
./eval.py solution.py first_autocorr
./eval.py solution.py heilbronn_triangle

# With fanout RSA (circle packing example)
fanout run "$(cat tasks/circle_packing.py)

Improve circle_packing26() to maximize sum of radii for 26 non-overlapping circles in the unit square. Output only the Python file." \
  -m openai/gpt-4o-mini -n 3 \
  -s rsa --k-agg 2 -r 5 \
  --eval-script "./eval.py" --materializer file --file-ext .py \
  -e script
```

## Scoring

Score = `benchmark_ratio` capped at 1.0, where `benchmark_ratio = your_metric / alphaevolve_metric`. A score of 1.0 means matching AlphaEvolve.
