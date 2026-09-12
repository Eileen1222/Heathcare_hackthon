"""Preprocessing utilities for the aorta-adjacent search region."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def make_aorta_shell(
    mask_np: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    outer_distance_mm: float = 10.0,
) -> np.ndarray:
    """Return voxels outside and within `outer_distance_mm` of the aorta."""
    aorta = np.asarray(mask_np) > 0
    if not np.any(aorta):
        raise ValueError("Aorta mask is empty")
    if outer_distance_mm <= 0:
        raise ValueError("outer_distance_mm must be positive")
    spacing_zyx = tuple(float(value) for value in reversed(spacing_xyz))
    distance = distance_transform_edt(~aorta, sampling=spacing_zyx)
    return (~aorta) & (distance <= outer_distance_mm)


def robust_intensity_window(
    image_np: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> tuple[float, float]:
    finite = np.asarray(image_np)[np.isfinite(image_np)]
    if finite.size == 0:
        raise ValueError("CTA contains no finite intensities")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    return float(low), float(high)
