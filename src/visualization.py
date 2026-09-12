"""Interactive 2D and 3D Plotly visualizations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from skimage.measure import marching_cubes


def create_slice_figure(image_np: np.ndarray, mask_np: np.ndarray, z: int):
    if not 0 <= z < image_np.shape[0]:
        raise IndexError(f"Slice {z} is outside [0, {image_np.shape[0] - 1}]")

    image_slice = np.asarray(image_np[z], dtype=float)
    mask_slice = np.asarray(mask_np[z], dtype=bool)
    window_min, window_max = np.percentile(image_slice, (1, 99))
    if window_min == window_max:
        window_max = window_min + 1.0

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            z=image_slice,
            colorscale="Gray",
            zmin=float(window_min),
            zmax=float(window_max),
            showscale=False,
            name="CT",
            hovertemplate="x=%{x}<br>y=%{y}<br>HU=%{z:.0f}<extra>CT</extra>",
        )
    )
    figure.add_trace(
        go.Heatmap(
            z=np.where(mask_slice, 1.0, np.nan),
            colorscale=[[0.0, "#ff7f0e"], [1.0, "#ff7f0e"]],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            opacity=0.4,
            name="Aorta mask",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        height=480,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        title={"text": f"Axial slice z={z}", "x": 0.5},
        dragmode="zoom",
        xaxis={"visible": False, "constrain": "domain"},
        yaxis={
            "visible": False,
            "autorange": "reversed",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    return figure


def create_aorta_figure(
    mask_np: np.ndarray,
    mask_image: sitk.Image,
    branches: Iterable[dict[str, Any]] = (),
) -> go.Figure:
    binary = np.asarray(mask_np) > 0
    if not np.any(binary):
        raise ValueError("Cannot visualize an empty aorta mask")

    vertices_zyx, faces, _, _ = marching_cubes(binary.astype(np.uint8), level=0.5)
    vertices_xyz_mm = _indices_to_physical(vertices_zyx, mask_image)
    figure = go.Figure(
        go.Mesh3d(
            x=vertices_xyz_mm[:, 0],
            y=vertices_xyz_mm[:, 1],
            z=vertices_xyz_mm[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="Aorta",
            color="#d94b64",
            opacity=0.55,
            flatshading=False,
        )
    )

    for branch in branches:
        ostium = np.asarray(branch["ostium_xyz_mm"], dtype=float)
        direction = np.asarray(branch["direction_xyz"], dtype=float)
        figure.add_trace(
            go.Scatter3d(
                x=[ostium[0]], y=[ostium[1]], z=[ostium[2]],
                mode="markers", marker={"size": 5, "color": "#ffd166"},
                name=str(branch["instance_id"]),
            )
        )
        figure.add_trace(
            go.Cone(
                x=[ostium[0]], y=[ostium[1]], z=[ostium[2]],
                u=[direction[0]], v=[direction[1]], w=[direction[2]],
                sizemode="absolute", sizeref=5, showscale=False,
                name=f"{branch['instance_id']} direction",
            )
        )

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (mm)",
            "yaxis_title": "y (mm)",
            "zaxis_title": "z (mm)",
        },
        legend={"orientation": "h"},
    )
    return figure


def _indices_to_physical(vertices_zyx: np.ndarray, image: sitk.Image) -> np.ndarray:
    indices_xyz = vertices_zyx[:, ::-1]
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return (direction @ (indices_xyz * spacing).T).T + origin
