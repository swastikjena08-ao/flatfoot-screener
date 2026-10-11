"""Shared constants and angle math for the bunion angle model."""
import math

import numpy as np

IN_W, IN_H = 256, 512  # model input (width, height); feet are about twice as tall as wide
STRIDE = 4             # heatmap downsampling
LINES = ("great_toe", "first_metatarsal", "second_metatarsal")
N_PTS = 6              # two endpoints per line, distal (toe end) first


def canonical_points(p):
    """Order each line's endpoints so the distal one (smaller y) comes first."""
    p = p.copy()
    for i in range(0, N_PTS, 2):
        if p[i, 1] > p[i + 1, 1]:
            p[[i, i + 1]] = p[[i + 1, i]]
    return p


def line_angle(a0, a1, b0, b1):
    """Acute angle in degrees between line a0-a1 and line b0-b1 (same as the dataset's points2angle)."""
    v1, v2 = np.subtract(a1, a0), np.subtract(b1, b0)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    t = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / n))))
    return min(t, 180 - t)


def angles(p):
    """HVA (big toe vs 1st metatarsal) and IMA (1st vs 2nd metatarsal) from 6 points."""
    hva = line_angle(p[0], p[1], p[2], p[3])
    ima = line_angle(p[2], p[3], p[4], p[5])
    return hva, ima
