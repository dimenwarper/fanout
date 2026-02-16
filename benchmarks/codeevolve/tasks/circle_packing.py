"""Circle Packing in a Unit Square (26 circles).

Objective: Place 26 non-overlapping circles inside the unit square [0,1]x[0,1]
to maximize the sum of their radii.

Benchmark (AlphaEvolve): sum_radii = 2.6358627564136983

Output: numpy array of shape (26, 3) where each row is (x, y, radius).
"""

import numpy as np

BENCHMARK_VALUE = 2.6358627564136983
NUM_CIRCLES = 26


def circle_packing26() -> np.ndarray:
    """Return (26, 3) array of (x, y, radius) for circles in unit square."""
    # Baseline: grid layout with uniform small circles
    n = int(np.ceil(np.sqrt(NUM_CIRCLES)))
    spacing = 1.0 / n
    radius = spacing / 2.5
    positions = np.zeros((NUM_CIRCLES, 3))
    idx = 0
    for i in range(n):
        for j in range(n):
            if idx >= NUM_CIRCLES:
                break
            positions[idx] = [spacing * (i + 0.5), spacing * (j + 0.5), radius]
            idx += 1
    return positions
