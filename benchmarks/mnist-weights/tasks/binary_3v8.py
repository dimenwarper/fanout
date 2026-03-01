"""Binary Classification: 3 vs 8 (harder pair).

Objective: Provide raw float weights for a tiny MLP that classifies digit 3
vs digit 8 from sklearn's load_digits (8x8 images). These digits are visually
similar, making this a harder task.

Architecture: 64 -> 8 (ReLU) -> 1 (sigmoid)  (529 parameters)
  - W1: (64, 8), b1: (8,)
  - W2: (8, 1), b2: (1,)

Benchmark: test accuracy >= 0.80

Output: dict with keys "W1", "b1", "W2", "b2" as numpy arrays.

IMPORTANT: You must produce the weight numbers directly — no training allowed.
Importing ML frameworks (sklearn, torch, tensorflow, etc.) or calling .fit()
will be detected and score 0. A 2-second time limit is also enforced.
Only numpy is allowed. Think about what pixel patterns distinguish 3 from 8
and encode that knowledge directly into the weight values.
"""

import numpy as np

BENCHMARK_VALUE = 0.80


def binary_3v8() -> dict:
    """Return weight dict for 64->8->1 binary MLP (3 vs 8)."""
    rng = np.random.RandomState(42)
    return {
        "W1": rng.randn(64, 8).astype(np.float32) * 0.1,
        "b1": np.zeros(8, dtype=np.float32),
        "W2": rng.randn(8, 1).astype(np.float32) * 0.1,
        "b2": np.zeros(1, dtype=np.float32),
    }
