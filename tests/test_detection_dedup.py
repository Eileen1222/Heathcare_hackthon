from __future__ import annotations

import unittest

import numpy as np

from src.detection import (
    DetectionConfig,
    _duplicate_evidence,
    _effective_centreline_distance_mm,
    _remove_duplicate_branches,
)


def _branch(
    *,
    component_id: int,
    ostium_zyx,
    path_zyx,
    radius_mm: float,
    ostium_radius_mm: float = 1.0,
    spacing=(1.0, 1.0, 1.0),
):
    path = np.asarray(path_zyx, dtype=float)
    ostium = np.asarray(ostium_zyx, dtype=float)
    seed = path[min(5, len(path) - 1)]
    spacing_arr = np.asarray(spacing, dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0) * spacing_arr, axis=1)
    path_length = float(lengths.sum()) if lengths.size else 0.0
    # Physical xyz uses the same numeric values for these unit-spacing tests.
    path_xyz = path[:, ::-1]
    return {
        "_component_id": component_id,
        "ostium_zyx": ostium.tolist(),
        "seed_zyx": seed.tolist(),
        "ostium_xyz_mm": tuple(ostium[::-1]),
        "seed_xyz_mm": tuple(seed[::-1]),
        "radius_mm": radius_mm,
        "ostium_radius_mm": ostium_radius_mm,
        "path_length_mm": path_length,
        "centreline_zyx": path.tolist(),
        "centreline_xyz_mm": [tuple(point) for point in path_xyz],
        "quality": {"cross_section_ratio": 1.1},
    }


class DetectionDedupTests(unittest.TestCase):
    def test_effective_distance_is_radius_aware(self):
        config = DetectionConfig()
        distance = _effective_centreline_distance_mm(0.4, 0.4, config)
        self.assertAlmostEqual(distance, 0.75 * 0.8, places=6)

    def test_nested_centrelines_are_merged(self):
        # Two nearly concentric tracks of the same lumen should collapse to one.
        path_a = [(0, 10, 10 + i) for i in range(11)]
        path_b = [(0.2, 10.1, 10 + i) for i in range(11)]
        branches = [
            _branch(
                component_id=1,
                ostium_zyx=(0, 10, 10),
                path_zyx=path_a,
                radius_mm=0.5,
                ostium_radius_mm=1.2,
            ),
            _branch(
                component_id=1,
                ostium_zyx=(0.2, 10.1, 10),
                path_zyx=path_b,
                radius_mm=0.5,
                ostium_radius_mm=0.9,
            ),
        ]
        image = np.full((8, 24, 24), 350.0, dtype=np.float32)
        kept, removed = _remove_duplicate_branches(
            branches, image, np.array([1.0, 1.0, 1.0]), DetectionConfig()
        )
        self.assertEqual(removed, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(
            kept[0]["quality"]["duplicate_check"]["status"],
            "representative_after_merge",
        )

    def test_nearby_but_diverging_origins_are_kept(self):
        # Ostia are close, but the paths split quickly into separate daughters.
        path_a = [(0, 10, 10 + i) for i in range(11)]
        path_b = [(0, 10 + i, 10 + i) for i in range(11)]
        branches = [
            _branch(
                component_id=1,
                ostium_zyx=(0, 10, 10),
                path_zyx=path_a,
                radius_mm=0.5,
            ),
            _branch(
                component_id=2,
                ostium_zyx=(0, 10.5, 10),
                path_zyx=path_b,
                radius_mm=0.5,
            ),
        ]
        image = np.full((8, 24, 24), 350.0, dtype=np.float32)
        kept, removed = _remove_duplicate_branches(
            branches, image, np.array([1.0, 1.0, 1.0]), DetectionConfig()
        )
        self.assertEqual(removed, 0)
        self.assertEqual(len(kept), 2)

    def test_intensity_gap_preserves_separate_vessels(self):
        path_a = [(5, 8, 10 + i) for i in range(11)]
        path_b = [(5, 12, 10 + i) for i in range(11)]
        first = _branch(
            component_id=1,
            ostium_zyx=(5, 8, 10),
            path_zyx=path_a,
            radius_mm=2.0,
        )
        second = _branch(
            component_id=2,
            ostium_zyx=(5, 12, 10),
            path_zyx=path_b,
            radius_mm=2.0,
        )
        image = np.full((12, 20, 24), 350.0, dtype=np.float32)
        # Low-HU slab between the two parallel tracks.
        image[:, 9:12, :] = 50.0
        evidence = _duplicate_evidence(
            first,
            second,
            image,
            np.array([1.0, 1.0, 1.0]),
            DetectionConfig(),
            same_component=False,
        )
        self.assertTrue(evidence["clear_intensity_gap"])
        self.assertFalse(evidence["possible_duplicate"])


if __name__ == "__main__":
    unittest.main()
