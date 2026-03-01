"""Top-5 Accuracy on 10-Class Digits.

Objective: Provide raw float weights for a tiny MLP where the correct label
is among the top 5 predictions. This is easier than top-1 accuracy.

Architecture: 64 -> 16 (ReLU) -> 10  (1,210 parameters)
  - W1: (64, 16), b1: (16,)
  - W2: (16, 10), b2: (10,)

Benchmark: top-5 test accuracy >= 0.85

Output: dict with keys "W1", "b1", "W2", "b2" as numpy arrays.

IMPORTANT: You must produce the weight numbers directly — no training allowed.
Importing ML frameworks (sklearn, torch, tensorflow, etc.) or calling .fit()
will be detected and score 0. A 2-second time limit is also enforced.
Only numpy is allowed. Think about what pixel patterns distinguish each digit
and encode that knowledge directly into the weight values.
"""

import numpy as np

BENCHMARK_VALUE = 0.85


def top5_accuracy() -> dict:
    """Return weight dict for 64->16->10 MLP (top-5 metric)."""
    rng = np.random.RandomState(42)
    return {
        "W1": rng.randn(64, 16).astype(np.float32) * 0.1,
        "b1": np.zeros(16, dtype=np.float32),
        "W2": rng.randn(16, 10).astype(np.float32) * 0.1,
        "b2": np.zeros(10, dtype=np.float32),
    }
