"""Generate Weights for Medium CNN on CIFAR-10.

Objective: Write code that produces weight arrays for a 3-conv CNN (~5K params).
Use mathematical patterns like Gabor filters, edge detectors, color templates.

Architecture (~5K params):
  Conv2d(3, 16, 3, padding=1) -> ReLU -> MaxPool(2)    # -> 16x16x16
  Conv2d(16, 32, 3, padding=1) -> ReLU -> MaxPool(2)   # -> 32x8x8
  Conv2d(32, 16, 3, padding=1) -> ReLU -> MaxPool(2)   # -> 16x4x4
  Linear(256, 10)

Weight dict keys:
  - "conv1.weight": (16, 3, 3, 3)
  - "conv1.bias": (16,)
  - "conv2.weight": (32, 16, 3, 3)
  - "conv2.bias": (32,)
  - "conv3.weight": (16, 32, 3, 3)
  - "conv3.bias": (16,)
  - "fc.weight": (10, 256)
  - "fc.bias": (10,)

Benchmark: test accuracy >= 0.20 on 2000 CIFAR-10 test images

Output: dict of numpy arrays with the keys above.
"""

import numpy as np

BENCHMARK_VALUE = 0.20


def generate_cnn_medium() -> dict:
    """Return weight dict for medium CNN on CIFAR-10."""
    rng = np.random.RandomState(42)
    return {
        "conv1.weight": rng.randn(16, 3, 3, 3).astype(np.float32) * 0.1,
        "conv1.bias": np.zeros(16, dtype=np.float32),
        "conv2.weight": rng.randn(32, 16, 3, 3).astype(np.float32) * 0.1,
        "conv2.bias": np.zeros(32, dtype=np.float32),
        "conv3.weight": rng.randn(16, 32, 3, 3).astype(np.float32) * 0.1,
        "conv3.bias": np.zeros(16, dtype=np.float32),
        "fc.weight": rng.randn(10, 256).astype(np.float32) * 0.1,
        "fc.bias": np.zeros(10, dtype=np.float32),
    }
