# CIFAR-10 WeightGen

The LLM writes Python code that generates weights for fixed CNN architectures on CIFAR-10. Instead of training, the LLM must craft weights using mathematical patterns — Gabor filters, edge detectors, color templates, etc.

## Architectures

**Small CNN** (~2K params):
```
Conv2d(3, 8, 3, padding=1) → ReLU → MaxPool(4)    # 8x8x8
Conv2d(8, 16, 3, padding=1) → ReLU → MaxPool(4)   # 16x2x2
Linear(64, 10)
```

**Medium CNN** (~5K params):
```
Conv2d(3, 16, 3, padding=1) → ReLU → MaxPool(2)    # 16x16x16
Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool(2)   # 32x8x8
Conv2d(32, 16, 3, padding=1) → ReLU → MaxPool(2)   # 16x4x4
Linear(256, 10)
```

**Linear only**: `Linear(3072, 10)`

## Tasks

| Task | Function | Architecture | Benchmark | Description |
|------|----------|-------------|-----------|-------------|
| `generate_cnn_small.py` | `generate_cnn_small()` | 2-conv CNN | accuracy >= 0.20 | Small CNN weight generation |
| `generate_cnn_medium.py` | `generate_cnn_medium()` | 3-conv CNN | accuracy >= 0.20 | Medium CNN weight generation |
| `generate_linear_only.py` | `generate_linear_only()` | Linear | accuracy >= 0.15 | Linear-only weight generation |

## Dependencies

`torch>=2.0`, `torchvision>=0.15`, `numpy>=1.26` (installed via `--extra benchmarks`)

CIFAR-10 auto-downloads to `.data/` on first run (~170MB).

## Eval

```bash
# Score a solution
python eval.py solution.py generate_cnn_small
python eval.py solution.py generate_linear_only
```

Evaluation uses a fixed subset of 2,000 CIFAR-10 test images (seed=42), CPU only.

## Run

```bash
# Single task, agent mode
uv run --extra benchmarks python benchmarks/cifar10-weightgen/run_benchmark.py \
  --tasks generate_cnn_small -m openai/gpt-4o-mini --mode agent --max-steps 3 -n 1

# All tasks
uv run --extra benchmarks python benchmarks/cifar10-weightgen/run_benchmark.py \
  -m openai/gpt-4o-mini --rounds 3

# Record results (higher eval timeout recommended)
uv run --extra benchmarks python benchmarks/cifar10-weightgen/run_benchmark.py \
  --tasks generate_cnn_small -m openai/gpt-4o-mini --mode agent --max-steps 5 -n 1 \
  --eval-timeout 120 --record my-run
```

## Scoring

Score is test accuracy (0 to 1) on the 2,000-image subset. Random chance is 0.10. Weight dict must match the architecture's expected keys and shapes exactly.

## CIFAR-10 Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
