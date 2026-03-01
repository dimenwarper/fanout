# MNIST-Weights

The LLM writes raw float weights for a tiny MLP that classifies sklearn's 8x8 digit images. No training — the LLM must directly produce weight values that perform well.

## Architecture

- **10-class tasks**: `64 → 16 (ReLU) → 10` — 1,210 parameters
- **Binary tasks**: `64 → 8 (ReLU) → 1 (sigmoid)` — 529 parameters

## Tasks

| Task | Function | Architecture | Benchmark | Description |
|------|----------|-------------|-----------|-------------|
| `classify_all.py` | `classify_all()` | 64→16→10 | accuracy >= 0.50 | 10-class digit classification |
| `binary_0v1.py` | `binary_0v1()` | 64→8→1 | accuracy >= 0.90 | Classify 0 vs 1 (easy pair) |
| `binary_3v8.py` | `binary_3v8()` | 64→8→1 | accuracy >= 0.80 | Classify 3 vs 8 (hard pair) |
| `top5_accuracy.py` | `top5_accuracy()` | 64→16→10 | top-5 acc >= 0.85 | Correct label in top 5 predictions |

## Dependencies

`scikit-learn>=1.4`, `numpy>=1.26` (installed via `--extra benchmarks`)

## Eval

```bash
# Score a solution
python eval.py solution.py classify_all
python eval.py solution.py binary_0v1
```

Data comes from `sklearn.datasets.load_digits` (8x8 grayscale, normalized to [0,1], 30% test split, seed=42).

## Run

```bash
# Single task, agent mode
uv run --extra benchmarks python benchmarks/mnist-weights/run_benchmark.py \
  --tasks binary_0v1 -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# All tasks
uv run --extra benchmarks python benchmarks/mnist-weights/run_benchmark.py \
  -m openai/gpt-4o-mini --rounds 3

# Record results
uv run --extra benchmarks python benchmarks/mnist-weights/run_benchmark.py \
  --tasks classify_all -m openai/gpt-4o-mini --mode agent --max-steps 5 -n 1 \
  --record my-run
```

## Anti-Cheat

Solutions must produce raw weight numbers directly — no training allowed:

- **Static analysis**: importing ML frameworks (sklearn, torch, tensorflow, keras, jax, etc.) or calling `.fit()` / `.train()` / `.backward()` is detected and scores 0 with an explanatory error
- **Time limit**: function must return within 2 seconds (returning numpy arrays takes milliseconds; training takes orders of magnitude longer)

Only `numpy` is allowed. The LLM must reason about digit pixel patterns and encode that knowledge directly into weight values.

## Scoring

Score is test accuracy (0 to 1). The forward pass is pure numpy — no frameworks needed in the solution. Weight dict must have keys `W1`, `b1`, `W2`, `b2` with correct shapes.
