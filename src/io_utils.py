"""NIfTI I/O, coordinate conversion, and challenge JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import SimpleITK as sitk


def load_image(path: str | Path) -> tuple[sitk.Image, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image, got {image.GetDimension()}D: {path}")
    return image, sitk.GetArrayFromImage(image)


def zyx_to_physical_xyz(
    image: sitk.Image, zyx: Sequence[float]
) -> tuple[float, float, float]:
    """Convert a NumPy `(z, y, x)` index to physical `(x, y, z)` mm."""
    if len(zyx) != 3:
        raise ValueError("zyx must contain exactly three coordinates")
    z, y, x = (float(value) for value in zyx)
    point = image.TransformContinuousIndexToPhysicalPoint((x, y, z))
    return tuple(float(value) for value in point)


def physical_xyz_to_zyx(
    image: sitk.Image, xyz_mm: Sequence[float]
) -> tuple[float, float, float]:
    """Convert physical `(x, y, z)` mm to a NumPy `(z, y, x)` index."""
    if len(xyz_mm) != 3:
        raise ValueError("xyz_mm must contain exactly three coordinates")
    x, y, z = image.TransformPhysicalPointToContinuousIndex(
        tuple(float(value) for value in xyz_mm)
    )
    return float(z), float(y), float(x)


def make_prediction(case_id: str, branches: Iterable[dict[str, Any]]) -> dict[str, Any]:
    daughters = []
    for index, branch in enumerate(branches, start=1):
        daughters.append(
            {
                "instance_id": str(branch.get("instance_id", f"branch_{index:03d}")),
                "parent_instance_id": "aorta",
                "ostium_xyz_mm": _float_triplet(branch["ostium_xyz_mm"]),
                "seed_xyz_mm": _float_triplet(branch["seed_xyz_mm"]),
                "radius_mm": float(branch["radius_mm"]),
                "direction_xyz": _unit_triplet(branch["direction_xyz"]),
            }
        )
    return {
        "case_id": str(case_id),
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }


def save_prediction(prediction: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(prediction, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _float_triplet(values: Sequence[float]) -> list[float]:
    if len(values) != 3:
        raise ValueError("Expected a three-value coordinate")
    result = [float(value) for value in values]
    if not np.all(np.isfinite(result)):
        raise ValueError("Coordinates must be finite")
    return result


def _unit_triplet(values: Sequence[float]) -> list[float]:
    vector = np.asarray(_float_triplet(values), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ValueError("Direction vector must be non-zero")
    return (vector / norm).tolist()
