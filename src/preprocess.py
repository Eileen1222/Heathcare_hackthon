"""Preprocessing utilities for the aorta-adjacent search region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class CroppedDistanceField:
    """Distance-to-aorta values stored only around the aorta."""

    volume_shape: tuple[int, int, int]
    roi: tuple[slice, slice, slice]
    values: np.ndarray
    spacing_zyx: tuple[float, float, float]

    def extract(self, target: tuple[slice, slice, slice]) -> np.ndarray:
        """Return a view matching a full-volume target ROI."""
        relative = []
        for target_slice, source_slice, size in zip(
            target, self.roi, self.volume_shape
        ):
            target_start = 0 if target_slice.start is None else target_slice.start
            target_stop = size if target_slice.stop is None else target_slice.stop
            source_start = 0 if source_slice.start is None else source_slice.start
            source_stop = size if source_slice.stop is None else source_slice.stop
            if target_start < source_start or target_stop > source_stop:
                raise ValueError("Requested ROI is outside the cached distance field")
            relative.append(
                slice(target_start - source_start, target_stop - source_start)
            )
        return self.values[tuple(relative)]


def make_aorta_search(
    mask_np: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    outer_distance_mm: float = 10.0,
    detection_halo_mm: float = 8.0,
) -> tuple[np.ndarray, CroppedDistanceField]:
    """Build the search shell and reusable distance field within a tight ROI."""
    aorta = np.asarray(mask_np) > 0
    if aorta.ndim != 3:
        raise ValueError("Aorta mask must be 3D")
    if not np.any(aorta):
        raise ValueError("Aorta mask is empty")
    if outer_distance_mm <= 0 or detection_halo_mm < 0:
        raise ValueError("Search distances must be positive")

    spacing_zyx_array = np.asarray(tuple(reversed(spacing_xyz)), dtype=float)
    if (
        spacing_zyx_array.shape != (3,)
        or np.any(~np.isfinite(spacing_zyx_array))
        or np.any(spacing_zyx_array <= 0)
    ):
        raise ValueError("Image spacing must contain three finite positive values")

    margin = (
        np.ceil(
            (outer_distance_mm + detection_halo_mm) / spacing_zyx_array
        ).astype(int)
        + 1
    )
    roi_parts = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(aorta.any(axis=other_axes))
        roi_parts.append(
            slice(
                max(0, int(occupied[0]) - int(margin[axis])),
                min(aorta.shape[axis], int(occupied[-1]) + int(margin[axis]) + 1),
            )
        )
    roi = tuple(roi_parts)
    aorta_roi = aorta[roi]
    distance_roi = distance_transform_edt(
        ~aorta_roi, sampling=tuple(spacing_zyx_array)
    )

    shell = np.zeros(aorta.shape, dtype=bool)
    shell[roi] = (~aorta_roi) & (distance_roi <= outer_distance_mm)
    distance_field = CroppedDistanceField(
        volume_shape=tuple(aorta.shape),
        roi=roi,
        values=distance_roi,
        spacing_zyx=tuple(float(value) for value in spacing_zyx_array),
    )
    return shell, distance_field


def make_aorta_shell(
    mask_np: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    outer_distance_mm: float = 10.0,
) -> np.ndarray:
    """Return voxels outside and within `outer_distance_mm` of the aorta."""
    shell, _ = make_aorta_search(
        mask_np,
        spacing_xyz,
        outer_distance_mm=outer_distance_mm,
    )
    return shell


def robust_intensity_window(
    image_np: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0
) -> tuple[float, float]:
    finite = np.asarray(image_np)[np.isfinite(image_np)]
    if finite.size == 0:
        raise ValueError("CTA contains no finite intensities")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    return float(low), float(high)
