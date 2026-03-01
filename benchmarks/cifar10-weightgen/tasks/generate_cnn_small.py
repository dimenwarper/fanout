"""Generate Weights for Small CNN on CIFAR-10.

Objective: Write code that produces weight arrays for a fixed small CNN
architecture, evaluated on CIFAR-10. The LLM should craft weights using
mathematical patterns (Gabor filters, edge detectors, color templates, etc.)
rather than training.

Architecture (~2K params):
  Conv2d(3, 8, 3, padding=1) -> ReLU -> MaxPool(4)    # -> 8x8x8
  Conv2d(8, 16, 3, padding=1) -> ReLU -> MaxPool(4)   # -> 16x2x2
  Linear(64, 10)

Weight dict keys:
  - "conv1.weight": (8, 3, 3, 3)
  - "conv1.bias": (8,)
  - "conv2.weight": (16, 8, 3, 3)
  - "conv2.bias": (16,)
  - "fc.weight": (10, 64)
  - "fc.bias": (10,)

Benchmark: test accuracy >= 0.20 on 2000 CIFAR-10 test images

Output: dict of numpy arrays with the keys above.
"""

import numpy as np

BENCHMARK_VALUE = 0.20


def generate_cnn_small() -> dict:
    """Return weight dict for small CNN on CIFAR-10."""
    rng = np.random.RandomState(42)
    return {
        "conv1.weight": rng.randn(8, 3, 3, 3).astype(np.float32) * 0.1,
        "conv1.bias": np.zeros(8, dtype=np.float32),
        "conv2.weight": rng.randn(16, 8, 3, 3).astype(np.float32) * 0.1,
        "conv2.bias": np.zeros(16, dtype=np.float32),
        "fc.weight": rng.randn(10, 64).astype(np.float32) * 0.1,
        "fc.bias": np.zeros(10, dtype=np.float32),
    }
