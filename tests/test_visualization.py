from __future__ import annotations

import unittest

import numpy as np
import SimpleITK as sitk

from src.visualization import create_aorta_figure


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
        )
        # Check that we have mesh3d traces for both aorta and the vessel tube
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        self.assertEqual(len(mesh_traces), 2)
        vessel_mesh = mesh_traces[1]
        self.assertEqual(vessel_mesh.name, "branch_001 vessel")
        self.assertGreater(len(vessel_mesh.i), 0)

    def test_aorta_figure_empty_branches(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((1.0, 1.0, 1.0))

        fig = create_aorta_figure(mask, mask_image, [])
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        self.assertEqual(len(mesh_traces), 1)
        self.assertEqual(mesh_traces[0].name, "Aorta")


if __name__ == "__main__":
    unittest.main()
