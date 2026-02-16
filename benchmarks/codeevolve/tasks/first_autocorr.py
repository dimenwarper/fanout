"""First Autocorrelation Inequality.

Objective: Find a sequence a of non-negative reals that minimizes the constant
C1 = 2n * max(b) / sum(a)^2, where b = convolve(a, a).

Equivalently, maximize inv_c1 = 1/C1.

Benchmark (AlphaEvolve): C1 < 1.5031 (inv_c1 > 0.6653)

Output: 1D numpy array of non-negative real values (the sequence a).
"""

import numpy as np

BENCHMARK_VALUE = 0.6653  # inv_c1


def first_autocorrelation() -> np.ndarray:
    """Return 1D array representing the sequence a."""
    # Baseline: uniform sequence
    n = 100
    a = np.ones(n)
    return a
