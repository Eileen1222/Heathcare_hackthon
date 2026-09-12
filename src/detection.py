"""Direct-daughter detection seam.

The stable return contract lets the detection teammate iterate independently.
Each returned branch must include the fields documented in `geometry.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import SimpleITK as sitk


def detect_branches(
    image: sitk.Image,
    image_np: np.ndarray,
    mask_np: np.ndarray,
    search_shell: np.ndarray,
) -> list[dict[str, Any]]:
    """Detect direct aortic daughters.

    Current scaffold returns no detections. Replace this body with the team's
    threshold/component or vessel-enhancement algorithm while preserving the
    return schema.
    """
    del image, image_np, mask_np, search_shell
    return []
