"""Geometry contract for branch results.

A completed branch dictionary contains:

    instance_id, ostium_zyx, seed_zyx,
    ostium_xyz_mm, seed_xyz_mm, radius_mm, direction_xyz

The `_zyx` fields are NumPy coordinates. The `_xyz_mm` fields and direction
are in the SimpleITK physical coordinate system.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def unit_direction_xyz(
    ostium_xyz_mm: Sequence[float], seed_xyz_mm: Sequence[float]
) -> tuple[float, float, float]:
    vector = np.asarray(seed_xyz_mm, dtype=float) - np.asarray(
        ostium_xyz_mm, dtype=float
    )
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Ostium and seed must be distinct")
    return tuple(float(value) for value in vector / norm)
