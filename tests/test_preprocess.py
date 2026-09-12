from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.preprocess import make_aorta_search, make_aorta_shell


class PreprocessTests(unittest.TestCase):
    def test_cropped_search_matches_full_volume_distance(self):
        mask = np.zeros((80, 80, 80), dtype=bool)
        mask[38:42, 38:42, 38:42] = True
        spacing_xyz = (0.8, 1.0, 1.2)
        spacing_zyx = spacing_xyz[::-1]

        shell, distance_field = make_aorta_search(
            mask,
            spacing_xyz,
            outer_distance_mm=5.0,
            detection_halo_mm=8.0,
        )
        full_distance = distance_transform_edt(~mask, sampling=spacing_zyx)
        expected_shell = (~mask) & (full_distance <= 5.0)

        np.testing.assert_array_equal(shell, expected_shell)
        np.testing.assert_allclose(
            distance_field.values,
            full_distance[distance_field.roi],
        )
        self.assertLess(distance_field.values.size, full_distance.size)

    def test_legacy_shell_api_uses_cropped_implementation(self):
        mask = np.zeros((40, 40, 40), dtype=bool)
        mask[18:22, 18:22, 18:22] = True

        shell = make_aorta_shell(mask, (1.0, 1.0, 1.0), outer_distance_mm=4.0)
        expected = (~mask) & (distance_transform_edt(~mask) <= 4.0)

        np.testing.assert_array_equal(shell, expected)


if __name__ == "__main__":
    unittest.main()
