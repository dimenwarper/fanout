"""Kissing Number in 11 dimensions.

Objective: Find the maximum number of integer-coordinate points in 11D space
such that max_norm <= min_pairwise_distance.

Benchmark (AlphaEvolve): 593 points

Output: numpy array of shape (N, 11) of integer coordinates.
"""

import numpy as np

BENCHMARK_VALUE = 593


def kissing_number11() -> np.ndarray:
    """Return (N, 11) array of integer points satisfying the kissing constraint."""
    d = 11
    # Baseline: just two antipodal points
    points = np.array([[1] * d, [-1] * d], dtype=int)
    return points
