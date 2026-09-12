"""The single core pipeline shared by the CLI and Streamlit app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.detection import detect_branches
from src.io_utils import load_image
from src.preprocess import make_aorta_search


def process_case(image_path: str | Path, mask_path: str | Path) -> dict[str, Any]:
    image, image_np = load_image(image_path)
    mask_image, mask_np = load_image(mask_path)
    _validate_pair(image, image_np, mask_image, mask_np)

    mask_np = mask_np > 0
    search_shell, distance_field = make_aorta_search(mask_np, image.GetSpacing())
    branches = detect_branches(
        image,
        image_np,
        mask_np,
        search_shell,
        distance_field=distance_field,
    )

    return {
        "image": image,
        "image_np": image_np,
        "mask_image": mask_image,
        "mask_np": mask_np,
        "search_shell": search_shell,
        "branches": branches,
    }


def _validate_pair(image, image_np, mask_image, mask_np) -> None:
    if image_np.shape != mask_np.shape:
        raise ValueError(
            f"CTA and mask shapes differ: {image_np.shape} vs {mask_np.shape}"
        )
    if not np.any(mask_np > 0):
        raise ValueError("Aorta mask is empty")
    if not np.allclose(image.GetSpacing(), mask_image.GetSpacing(), rtol=1e-3, atol=1e-3):
        raise ValueError("CTA and mask spacing differ")
    if not np.allclose(image.GetOrigin(), mask_image.GetOrigin(), rtol=1e-3, atol=1e-2):
        raise ValueError("CTA and mask origins differ")
    if not np.allclose(image.GetDirection(), mask_image.GetDirection(), rtol=1e-3, atol=1e-3):
        raise ValueError("CTA and mask directions differ")
