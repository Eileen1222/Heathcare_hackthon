from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.visualization import (
    create_aorta_figure,
    create_slice_figure,
    prepare_aorta_3d_geometry,
)


class VisualizationTests(unittest.TestCase):
    def test_aorta_figure_generates_3d_vessel_tubes(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((1.0, 1.0, 1.0))

        branches = [
            {
                "instance_id": "branch_001",
                "ostium_xyz_mm": (10.0, 10.0, 15.0),
                "seed_xyz_mm": (10.0, 10.0, 20.0),
                "radius_mm": 1.5,
                "direction_xyz": (0.0, 0.0, 1.0),
                "centreline_xyz_mm": [
                    (10.0, 10.0, float(z)) for z in range(15, 25)
                ],
                "path_length_mm": 9.0,
            }
        ]

        fig = create_aorta_figure(
            mask,
            mask_image,
            branches,
            show_vessels=True,
            show_centerlines=True,
            focused_branch_id="branch_001",
            slice_plane_info=("axial", 10),
        )
        # Check that we have mesh3d traces for aorta, vessel tube, and slice plane
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        self.assertEqual(len(mesh_traces), 3)
        self.assertEqual(mesh_traces[0].name, "Aorta")
        self.assertIn("branch_001", mesh_traces[1].name)
        self.assertIn("Axial plane", mesh_traces[2].name)

    def test_slice_figure_focuses_branch(self):
        image_np = np.zeros((20, 20, 20), dtype=np.int16)
        mask_np = np.zeros((20, 20, 20), dtype=np.uint8)
        mask_np[5:15, 5:15, 5:15] = 1
        branches = [
            {
                "instance_id": "branch_001",
                "ostium_zyx": (10.0, 10.0, 10.0),
                "seed_zyx": (10.0, 12.0, 10.0),
                "radius_mm": 1.2,
                "direction_xyz": (0.0, 1.0, 0.0),
                "centreline_zyx": [(10.0, float(y), 10.0) for y in range(10, 15)],
            }
        ]
        fig = create_slice_figure(
            image_np,
            mask_np,
            10,
            branches=branches,
            plane="axial",
            focused_branch_id="branch_001",
        )
        self.assertGreater(len(fig.data), 0)

    def test_aorta_figure_empty_branches(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((1.0, 1.0, 1.0))

        fig = create_aorta_figure(mask, mask_image, [])
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        self.assertEqual(len(mesh_traces), 1)
        self.assertEqual(mesh_traces[0].name, "Aorta")

    def test_prepared_geometry_skips_mesh_recomputation(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        prepared = prepare_aorta_3d_geometry(mask, mask_image, [])

        with patch(
            "src.visualization.marching_cubes",
            side_effect=AssertionError("mesh should be reused"),
        ):
            figure = create_aorta_figure(
                mask,
                mask_image,
                [],
                prepared_geometry=prepared,
                slice_plane_info=("axial", 10),
            )

        self.assertEqual(figure.data[0].name, "Aorta")


if __name__ == "__main__":
    unittest.main()
