"""10-Class Digit Classification via Raw Weights.

Objective: Provide raw float weights for a tiny MLP that classifies all 10
digits from sklearn's load_digits (8x8 images, 64 input features).

Architecture: 64 -> 16 (ReLU) -> 10  (1,210 parameters)
  - W1: (64, 16), b1: (16,)
  - W2: (16, 10), b2: (10,)

Benchmark: test accuracy >= 0.50

Output: dict with keys "W1", "b1", "W2", "b2" as numpy arrays.

IMPORTANT: You must produce the weight numbers directly — no training allowed.
Importing ML frameworks (sklearn, torch, tensorflow, etc.) or calling .fit()
will be detected and score 0. Dynamic imports (__import__, importlib) and
loading the eval dataset (load_digits) are also banned. A 0.5-second time
limit is enforced — no training loops, even in raw numpy.
Only numpy is allowed. Think about what pixel patterns distinguish each digit
and encode that knowledge directly into the weight values.
"""

import numpy as np

BENCHMARK_VALUE = 0.50


def classify_all() -> dict:
    """Return weight dict for 64->16->10 MLP."""
    rng = np.random.RandomState(42)
    return {
        "W1": rng.randn(64, 16).astype(np.float32) * 0.1,
        "b1": np.zeros(16, dtype=np.float32),
        "W2": rng.randn(16, 10).astype(np.float32) * 0.1,
        "b2": np.zeros(10, dtype=np.float32),
    }
