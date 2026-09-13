from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from src.visualization import (
    CLINICAL_OSTIUM,
    CLINICAL_SEED,
    CLINICAL_VESSEL,
    _HAS_PYVISTA,
    _build_tube_mesh,
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
        # Rim + lumen shells, selected/branch tube, and slice plane.
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        mesh_names = [t.name for t in mesh_traces]
        self.assertIn("Vessel rim", mesh_names)
        self.assertIn("Vessel lumen", mesh_names)
        self.assertIn("Selected branch", mesh_names)
        self.assertTrue(any("Axial plane" in name for name in mesh_names))
        rim = next(t for t in mesh_traces if t.name == "Vessel rim")
        lumen = next(t for t in mesh_traces if t.name == "Vessel lumen")
        self.assertEqual(rim.color, "#9a0000")
        self.assertEqual(lumen.color, "#ff7a7a")
        self.assertLess(lumen.opacity, rim.opacity)
        # Mesh3d carries branch hover text (Scatter inside mesh cannot receive hover).
        rim_text = list(rim.hovertext or rim.text or [])
        self.assertTrue(any("branch_001" in str(t) for t in rim_text))
        self.assertTrue(any("Radius" in str(t) for t in rim_text))
        selected = next(t for t in mesh_traces if t.name == "Selected branch")
        self.assertIsNotNone(selected.hovertemplate)
        self.assertIn("Radius", selected.hovertemplate)
        self.assertTrue(
            any(
                t.type == "scatter3d"
                and t.name == "Ostium"
                and t.marker.color == CLINICAL_OSTIUM
                for t in fig.data
            )
        )
        self.assertFalse(
            any(getattr(t, "name", None) == "branch_001 hover" for t in fig.data)
        )
        self.assertEqual(getattr(lumen, "hoverinfo", None), "none")

    def test_aorta_figure_uses_unified_vessel_colour(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((1.0, 1.0, 1.0))
        branches = [
            {
                "instance_id": f"branch_{i:03d}",
                "ostium_xyz_mm": (10.0, 10.0, 12.0 + i),
                "seed_xyz_mm": (10.0, 10.0, 18.0 + i),
                "radius_mm": 1.0,
                "direction_xyz": (0.0, 0.0, 1.0),
                "centreline_xyz_mm": [
                    (10.0, 10.0, float(z)) for z in range(12 + i, 20 + i)
                ],
                "path_length_mm": 8.0,
            }
            for i in range(1, 4)
        ]
        fig = create_aorta_figure(
            mask, mask_image, branches, show_vessels=True, show_markers=True
        )
        tree_meshes = [
            t
            for t in fig.data
            if t.type == "mesh3d" and t.name in {"Vessel rim", "Vessel lumen"}
        ]
        self.assertEqual(len(tree_meshes), 2)
        self.assertEqual(CLINICAL_VESSEL, "#ff0000")
        prepared = prepare_aorta_3d_geometry(mask, mask_image, branches)
        self.assertIn("tree_rim_vertices_xyz_mm", prepared)
        self.assertIn("tree_inner_vertices_xyz_mm", prepared)
        self.assertGreater(len(prepared["tree_vertices_xyz_mm"]), 0)
        label_count = sum(
            1 for t in fig.data if getattr(t, "name", None) == "Branch labels"
        )
        self.assertEqual(label_count, 3)

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
        ostium_traces = [
            t for t in fig.data if getattr(t, "name", None) == "branch_001"
        ]
        self.assertTrue(ostium_traces)
        self.assertEqual(ostium_traces[0].marker.color, CLINICAL_OSTIUM)

    def test_aorta_figure_empty_branches(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mask_image = sitk.GetImageFromArray(mask)
        mask_image.SetSpacing((1.0, 1.0, 1.0))

        fig = create_aorta_figure(mask, mask_image, [])
        mesh_traces = [t for t in fig.data if t.type == "mesh3d"]
        mesh_names = {t.name for t in mesh_traces}
        self.assertTrue({"Vessel rim", "Vessel lumen"} <= mesh_names)

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

        self.assertEqual(figure.data[0].name, "Vessel rim")
        self.assertEqual(figure.data[1].name, "Vessel lumen")

    def test_tube_mesh_uses_pyvista_and_ostium_flare(self):
        path = [(0.0, 0.0, float(z)) for z in range(0, 12)]
        mesh = _build_tube_mesh(
            path,
            radius=1.0,
            ostium_radius_mm=2.5,
            n_radial=20,
        )
        self.assertIsNotNone(mesh)
        verts, i_idx, j_idx, k_idx = mesh
        self.assertGreater(len(verts), 40)
        self.assertEqual(len(i_idx), len(j_idx))
        self.assertEqual(len(j_idx), len(k_idx))
        self.assertGreater(len(i_idx), 20)
        # Distal tip remains an open pipe (no sealed center vertex on the tip).
        # Proximal aorta-side rings are trimmed for a smooth junction.
        self.assertGreater(float(verts[:, 2].max()), 10.0)
        # Open pipe at distal tip only (proximal aorta junction has no pipe-mouth lip).
        self.assertGreaterEqual(len(verts), 50)
        path = np.asarray(path, dtype=float)
        distal = path[-1]
        dists = np.linalg.norm(verts - distal, axis=1)
        self.assertGreater(float(dists.min()), 0.15)

    def test_short_branch_tube_is_lengthened_not_ring_stack(self):
        # Ostium→seed only ~2 mm: previously looked like stacked washers.
        mesh = _build_tube_mesh(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)],
            radius=0.7,
            n_radial=16,
        )
        self.assertIsNotNone(mesh)
        verts, i_idx, _, _ = mesh
        span = float(verts[:, 2].max() - verts[:, 2].min())
        self.assertGreaterEqual(span, 6.0)
        # Adaptive sampling + single outer wall: far fewer verts than dual 48-ring tubes.
        self.assertLess(len(verts), 500)
        self.assertGreater(len(i_idx), 20)

    def test_tube_keeps_distal_open_lip_without_proximal_pipe_mouth(self):
        mesh = _build_tube_mesh(
            [(0.0, 0.0, float(z)) for z in range(0, 14)],
            radius=1.0,
            n_radial=16,
        )
        self.assertIsNotNone(mesh)
        verts, _, _, _ = mesh
        # Distal lip adds an inset ring → more verts than a pure outer shell of ~N*16.
        # Proximal cut removes aorta-side rings, so total stays moderate.
        self.assertGreater(len(verts), 16 * 4)
        self.assertLess(len(verts), 16 * 40)


if __name__ == "__main__":
    unittest.main()
