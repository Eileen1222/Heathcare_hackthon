from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.io_utils import make_prediction, save_prediction
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

    def test_empty_mask_raises_value_error(self):
        image_np = np.zeros((16, 16, 16), dtype=np.int16)
        mask_np = np.zeros((16, 16, 16), dtype=np.uint8)
        image = sitk.GetImageFromArray(image_np)
        mask_image = sitk.GetImageFromArray(mask_np)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            sitk.WriteImage(image, str(image_path))
            sitk.WriteImage(mask_image, str(mask_path))
            with self.assertRaises(ValueError) as ctx:
                process_case(image_path, mask_path)
            self.assertIn("Aorta mask is empty", str(ctx.exception))

    def test_shape_mismatch_raises_value_error(self):
        image = sitk.GetImageFromArray(np.zeros((16, 16, 16), dtype=np.int16))
        mask_image = sitk.GetImageFromArray(np.ones((16, 16, 20), dtype=np.uint8))

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.nii.gz"
            mask_path = Path(directory) / "mask.nii.gz"
            sitk.WriteImage(image, str(image_path))
            sitk.WriteImage(mask_image, str(mask_path))
            with self.assertRaises(ValueError) as ctx:
                process_case(image_path, mask_path)
            self.assertIn("shapes differ", str(ctx.exception))

    def test_zero_branches_prediction_validity(self):
        pred = make_prediction("empty_case", [])
        self.assertEqual(pred["case_id"], "empty_case")
        self.assertEqual(pred["parent"]["instance_id"], "aorta")
        self.assertEqual(pred["daughters"], [])

        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory) / "empty.json"
            save_prediction(pred, out_path)
            self.assertTrue(out_path.is_file())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn('"daughters": []', content)

    def test_cli_handles_missing_file_cleanly(self):
        ret = subprocess.run(
            [
                sys.executable,
                "run.py",
                "--image",
                "nonexistent_image.nii.gz",
                "--aorta-mask",
                "nonexistent_mask.nii.gz",
                "--output",
                "output.json",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(ret.returncode, 2)
        self.assertIn("File Not Found", ret.stderr)


if __name__ == "__main__":
    unittest.main()
