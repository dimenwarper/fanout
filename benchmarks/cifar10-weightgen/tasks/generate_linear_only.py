"""Generate Weights for Linear-Only Model on CIFAR-10.

Objective: Write code that produces weight arrays for a single linear layer
that classifies flattened CIFAR-10 images (3072 -> 10). This is the simplest
possible architecture -- craft weights using color/spatial templates.

Architecture:
  Linear(3072, 10)

Weight dict keys:
  - "fc.weight": (10, 3072)
  - "fc.bias": (10,)

Benchmark: test accuracy >= 0.15 on 2000 CIFAR-10 test images

Output: dict of numpy arrays with the keys above.
"""

import numpy as np

BENCHMARK_VALUE = 0.15


def generate_linear_only() -> dict:
    """Return weight dict for linear model on CIFAR-10."""
    rng = np.random.RandomState(42)
    return {
        "fc.weight": rng.randn(10, 3072).astype(np.float32) * 0.01,
        "fc.bias": np.zeros(10, dtype=np.float32),
    }
