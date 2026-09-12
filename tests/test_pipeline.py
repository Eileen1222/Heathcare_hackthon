from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.pipeline import process_case


class PipelineTests(unittest.TestCase):
    def test_synthetic_case_runs_end_to_end(self):
        zz, yy, xx = np.ogrid[:32, :32, :32]
        mask = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 6**2).astype(np.uint8)
        mask = np.broadcast_to(mask, (32, 32, 32)).copy()
        ct = (mask * 250).astype(np.int16)
        image = sitk.GetImageFromArray(ct)
        mask_image = sitk.GetImageFromArray(mask)
        image.SetSpacing((0.8, 0.8, 1.2))
        mask_image.CopyInformation(image)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            sitk.WriteImage(image, str(image_path))
            sitk.WriteImage(mask_image, str(mask_path))
            result = process_case(image_path, mask_path)

        self.assertEqual(result["image_np"].shape, (32, 32, 32))
        self.assertTrue(result["search_shell"].any())
        self.assertEqual(result["branches"], [])


if __name__ == "__main__":
    unittest.main()
