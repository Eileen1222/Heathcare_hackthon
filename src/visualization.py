"""Interactive 2D and 3D Plotly visualizations."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from skimage.measure import marching_cubes


def create_slice_figure(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    slice_index: int,
    branches: Iterable[dict[str, Any]] = (),
    *,
    plane: str = "axial",
    spacing_zyx: Iterable[float] = (1.0, 1.0, 1.0),
    show_mask: bool = True,
    show_points: bool = True,
    show_centerlines: bool = True,
    show_radius: bool = True,
) -> go.Figure:
    """Create an interactive CT slice with physically scaled detection overlays."""
    plane_specs = {
        "axial": (0, 1, 2, "z"),
        "coronal": (1, 0, 2, "y"),
        "sagittal": (2, 0, 1, "x"),
    }
    if plane not in plane_specs:
        raise ValueError(f"Unknown plane: {plane}")

    fixed_axis, row_axis, column_axis, axis_name = plane_specs[plane]
    if not 0 <= slice_index < image_np.shape[fixed_axis]:
        raise IndexError(
            f"Slice {slice_index} is outside "
            f"[0, {image_np.shape[fixed_axis] - 1}] for {plane}"
        )

    spacing = np.asarray(tuple(spacing_zyx), dtype=float)
    if spacing.shape != (3,) or np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_zyx must contain three finite positive values")

    image_slice = np.asarray(
        np.take(image_np, slice_index, axis=fixed_axis), dtype=float
    )
    mask_slice = np.asarray(
        np.take(mask_np, slice_index, axis=fixed_axis), dtype=bool
    )
    x_coordinates = np.arange(image_slice.shape[1]) * spacing[column_axis]
    y_coordinates = np.arange(image_slice.shape[0]) * spacing[row_axis]
    window_min, window_max = np.percentile(image_slice, (1, 99))
    if window_min == window_max:
        window_max = window_min + 1.0

    figure = go.Figure()
    figure.add_trace(
        go.Heatmap(
            x=x_coordinates,
            y=y_coordinates,
            z=image_slice,
            colorscale="Gray",
            zmin=float(window_min),
            zmax=float(window_max),
            showscale=False,
            name="CT",
            hovertemplate=(
                "horizontal=%{x:.1f} mm<br>vertical=%{y:.1f} mm"
                "<br>HU=%{z:.0f}<extra>CT</extra>"
            ),
        )
    )
    if show_mask:
        figure.add_trace(
            go.Heatmap(
                x=x_coordinates,
                y=y_coordinates,
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

    tolerance_mm = max(1.0, 0.75 * spacing[fixed_axis])
    branch_colors = (
        "#00d4ff",
        "#7bed9f",
        "#ffd166",
        "#ff6b9d",
        "#a78bfa",
        "#45e0a8",
    )
    for branch_index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{branch_index + 1:03d}"))
        color = branch_colors[branch_index % len(branch_colors)]
        legend_shown = False

        if show_centerlines and branch.get("centreline_zyx"):
            centreline = np.asarray(branch["centreline_zyx"], dtype=float)
            near_slice = (
                np.abs(centreline[:, fixed_axis] - slice_index)
                * spacing[fixed_axis]
                <= tolerance_mm
            )
            line_x: list[float | None] = []
            line_y: list[float | None] = []
            for point, is_near in zip(centreline, near_slice):
                if is_near:
                    line_x.append(float(point[column_axis] * spacing[column_axis]))
                    line_y.append(float(point[row_axis] * spacing[row_axis]))
                elif line_x and line_x[-1] is not None:
                    line_x.append(None)
                    line_y.append(None)
            if any(value is not None for value in line_x):
                figure.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines+markers",
                        line={"color": color, "width": 3},
                        marker={"color": color, "size": 3},
                        name=branch_id,
                        legendgroup=branch_id,
                        showlegend=True,
                        hovertemplate=f"{branch_id} centerline<extra></extra>",
                    )
                )
                legend_shown = True

        if not show_points:
            continue

        ostium = np.asarray(branch["ostium_zyx"], dtype=float)
        seed = np.asarray(branch["seed_zyx"], dtype=float)
        ostium_near = (
            abs(ostium[fixed_axis] - slice_index) * spacing[fixed_axis]
            <= tolerance_mm
        )
        seed_near = (
            abs(seed[fixed_axis] - slice_index) * spacing[fixed_axis]
            <= tolerance_mm
        )
        ostium_x = float(ostium[column_axis] * spacing[column_axis])
        ostium_y = float(ostium[row_axis] * spacing[row_axis])
        seed_x = float(seed[column_axis] * spacing[column_axis])
        seed_y = float(seed[row_axis] * spacing[row_axis])

        if ostium_near:
            figure.add_trace(
                go.Scatter(
                    x=[ostium_x],
                    y=[ostium_y],
                    mode="markers+text",
                    marker={
                        "size": 11,
                        "color": "#ffe066",
                        "line": {"color": color, "width": 2},
                    },
                    text=[branch_id],
                    textposition="top center",
                    textfont={"color": color},
                    name=branch_id,
                    legendgroup=branch_id,
                    showlegend=not legend_shown,
                    hovertemplate=(
                        f"{branch_id}<br>ostium"
                        f"<br>z/y/x={ostium[0]:.1f}/{ostium[1]:.1f}/{ostium[2]:.1f}"
                        "<extra></extra>"
                    ),
                )
            )
            legend_shown = True
            # The arrow is the projection of the 3D ostium-to-seed direction
            # into the selected viewing plane.
            figure.add_annotation(
                x=seed_x,
                y=seed_y,
                ax=ostium_x,
                ay=ostium_y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=3,
                arrowcolor=color,
                text="",
            )

        if seed_near or ostium_near:
            is_projection = not seed_near
            figure.add_trace(
                go.Scatter(
                    x=[seed_x],
                    y=[seed_y],
                    mode="markers",
                    marker={
                        "size": 9,
                        "symbol": "x" if is_projection else "diamond",
                        "color": color,
                    },
                    name=f"{branch_id} seed",
                    legendgroup=branch_id,
                    showlegend=False,
                    hovertemplate=(
                        f"{branch_id}<br>"
                        + ("seed projection" if is_projection else "5 mm seed")
                        + f"<br>radius={float(branch['radius_mm']):.2f} mm"
                        + "<extra></extra>"
                    ),
                )
            )

        if show_radius and seed_near:
            radius = float(branch["radius_mm"])
            figure.add_shape(
                type="circle",
                x0=seed_x - radius,
                x1=seed_x + radius,
                y0=seed_y - radius,
                y1=seed_y + radius,
                line={"color": color, "width": 2, "dash": "dot"},
            )

    figure.update_layout(
        height=430,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        title={
            "text": f"{plane.capitalize()} slice {axis_name}={slice_index}",
            "x": 0.5,
        },
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
