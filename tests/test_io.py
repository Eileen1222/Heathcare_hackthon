from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.io_utils import (
    load_image,
    make_prediction,
    physical_xyz_to_zyx,
    zyx_to_physical_xyz,
)


class IoTests(unittest.TestCase):
    def test_nifti_round_trip_and_coordinate_conversion(self):
        array = np.zeros((8, 9, 10), dtype=np.int16)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 1.5))
        image.SetOrigin((10.0, -20.0, 30.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.nii.gz"
            sitk.WriteImage(image, str(path))
            loaded_image, loaded_array = load_image(path)
        np.testing.assert_array_equal(loaded_array, array)
        zyx = (3.0, 4.0, 5.0)
        xyz_mm = zyx_to_physical_xyz(loaded_image, zyx)
        np.testing.assert_allclose(physical_xyz_to_zyx(loaded_image, xyz_mm), zyx)

    def test_prediction_normalizes_direction(self):
        prediction = make_prediction(
            "case-1",
            [{
                "instance_id": "branch_001",
                "ostium_xyz_mm": (1, 2, 3),
                "seed_xyz_mm": (6, 2, 3),
                "radius_mm": 2.5,
                "direction_xyz": (5, 0, 0),
            }],
        )
        self.assertEqual(prediction["daughters"][0]["direction_xyz"], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
