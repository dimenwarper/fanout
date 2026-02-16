"""Heilbronn Triangle Problem (11 points).

Objective: Place 11 points inside an equilateral triangle with vertices at
(0,0), (1,0), (0.5, sqrt(3)/2) to maximize the minimum triangle area
among all C(11,3) = 165 possible triangles.

Benchmark (AlphaEvolve): min_area_normalized = 0.036529889880030156

Output: numpy array of shape (11, 2) with (x, y) coordinates.
"""

import numpy as np

BENCHMARK_VALUE = 0.036529889880030156
NUM_POINTS = 11


def heilbronn_triangle11() -> np.ndarray:
    """Return (11, 2) array of point positions inside the equilateral triangle."""
    # Baseline: evenly spaced points along the centroid line
    h = np.sqrt(3) / 2
    points = np.zeros((NUM_POINTS, 2))
    for i in range(NUM_POINTS):
        t = (i + 1) / (NUM_POINTS + 1)
        points[i] = [0.5, t * h]
    return points
