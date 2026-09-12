"""Interactive 2D and 3D Plotly visualizations."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from skimage.measure import marching_cubes

BRANCH_COLORS: tuple[str, ...] = (
    "#00d4ff",  # 电光青
    "#7bed9f",  # 薄荷绿
    "#ffd166",  # 琥珀黄
    "#ff6b9d",  # 珊瑚粉
    "#a78bfa",  # 丁香紫
    "#ffa502",  # 暖橙
    "#48dbfb",  # 冰天蓝
    "#2ed573",  # 翡翠绿
    "#ff4757",  # 活力红
    "#1e90ff",  # 道奇蓝
    "#eccc68",  # 暖金
    "#f368e0",  # 桃红
)


def create_slice_figure(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    slice_index: int,
    branches: Iterable[dict[str, Any]] = (),
    *,
    plane: str = "axial",
    spacing_zyx: Iterable[float] = (1.0, 1.0, 1.0),
    show_mask: bool = True,
    show_ostia: bool = True,
    show_seeds: bool = True,
    show_directions: bool = True,
    show_centerlines: bool = True,
    show_radius: bool = True,
    **kwargs: Any,
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
    # Each pair contains a saturated branch/direction color and a lighter
    # same-family centerline color.
    branch_palette = (
        ("#00a8cc", "#7ee8fa"),
        ("#2eae67", "#9cf2c0"),
        ("#e2a400", "#ffe49a"),
        ("#d94f87", "#ffadd0"),
        ("#805ad5", "#c4b5fd"),
        ("#008f72", "#7de2cb"),
    )
    for branch_index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{branch_index + 1:03d}"))
        color, centerline_color = branch_palette[branch_index % len(branch_palette)]
        # A legend-only marker keeps the branch list stable even when the
        # selected slice does not intersect that branch.
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 9, "color": color},
                name=branch_id,
                legendgroup=branch_id,
                showlegend=True,
                hoverinfo="skip",
            )
        )

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
                        line={"color": centerline_color, "width": 3, "dash": "dot"},
                        marker={"color": centerline_color, "size": 3},
                        name=f"{branch_id} centerline",
                        legendgroup=branch_id,
                        showlegend=False,
                        hovertemplate=f"{branch_id} centerline<extra></extra>",
                    )
                )

        if not (show_ostia or show_seeds or show_directions or show_radius):
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

        if show_ostia and ostium_near:
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
                    showlegend=False,
                    hovertemplate=(
                        f"{branch_id}<br>ostium"
                        f"<br>z/y/x={ostium[0]:.1f}/{ostium[1]:.1f}/{ostium[2]:.1f}"
                        "<extra></extra>"
                    ),
                )
            )

        if show_directions and ostium_near:
            # The arrow is the projection of the 3D ostium-to-seed direction
            # into the selected viewing plane.
            figure.add_trace(
                go.Scatter(
                    x=[ostium_x, seed_x],
                    y=[ostium_y, seed_y],
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                    marker={
                        "color": color,
                        "size": [0, 12],
                        "symbol": ["circle", "arrow"],
                        "angleref": "previous",
                    },
                    name=f"{branch_id} direction",
                    legendgroup=branch_id,
                    showlegend=False,
                    hovertemplate=f"{branch_id} direction<extra></extra>",
                )
            )

        if show_seeds and (seed_near or (show_directions and ostium_near)):
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
            angles = np.linspace(0.0, 2.0 * np.pi, 65)
            figure.add_trace(
                go.Scatter(
                    x=seed_x + radius * np.cos(angles),
                    y=seed_y + radius * np.sin(angles),
                    mode="lines",
                    line={"color": color, "width": 2, "dash": "dot"},
                    name=f"{branch_id} radius",
                    legendgroup=branch_id,
                    showlegend=False,
                    hovertemplate=(
                        f"{branch_id}<br>radius={radius:.2f} mm<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        height=580,
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
        legend={
            "title": {"text": "Branches"},
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(14, 17, 23, 0.72)",
            "bordercolor": "rgba(255, 255, 255, 0.25)",
            "borderwidth": 1,
            "font": {"color": "#fafafa"},
            "groupclick": "togglegroup",
        },
    )
    return figure


def _resample_or_smooth_path(points: np.ndarray, num_points: int = 24) -> np.ndarray:
    """Resample center line points with cumulative arc-length for smooth tube generation."""
    if len(points) <= 2:
        return points
    diffs = np.diff(points, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    valid_idx = np.ones(len(points), dtype=bool)
    valid_idx[1:] = step_dists > 1e-4
    pts = points[valid_idx]
    if len(pts) <= 2:
        return pts

    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_len = cum_dist[-1]
    if total_len < 1e-4:
        return pts[:2]

    target_count = max(8, min(num_points, len(pts) * 2))
    target_dists = np.linspace(0.0, total_len, target_count)
    resampled = np.zeros((len(target_dists), 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_dists, cum_dist, pts[:, dim])
    return resampled


def _build_tube_mesh(
    points: Sequence[Sequence[float]],
    radius: float,
    n_radial: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a 3D tubular mesh (vertices, I, J, K) along a 3D centerline path.

    Uses a parallel-transport (Bishop) frame along the curve to prevent
    artificial twisting and pinching around bends.
    """
    raw_pts = np.asarray(points, dtype=float)
    if len(raw_pts) < 2:
        return None

    pts = _resample_or_smooth_path(raw_pts)
    if len(pts) < 2:
        return None

    tangents = np.zeros_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    tangents /= norms

    t0 = tangents[0]
    ref_up = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t0, ref_up)
    n0_norm = np.linalg.norm(n0)
    if n0_norm < 1e-6:
        n0 = np.cross(t0, np.array([1.0, 0.0, 0.0]))
        n0_norm = np.linalg.norm(n0)
    n0 /= max(n0_norm, 1e-8)
    b0 = np.cross(t0, n0)

    normals = [n0]
    binormals = [b0]

    for i in range(1, len(pts)):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        axis = np.cross(t_prev, t_curr)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            ni = normals[-1]
            ni = ni - np.dot(ni, t_curr) * t_curr
            ni_norm = np.linalg.norm(ni)
            ni = ni / max(ni_norm, 1e-8)
        else:
            axis_unit = axis / axis_len
            cos_theta = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            v = normals[-1]
            ni = (
                v * np.cos(theta)
                + np.cross(axis_unit, v) * np.sin(theta)
                + axis_unit * np.dot(axis_unit, v) * (1.0 - np.cos(theta))
            )
            ni = ni / max(np.linalg.norm(ni), 1e-8)
        bi = np.cross(t_curr, ni)
        normals.append(ni)
        binormals.append(bi)

    angles = np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    vertices_list = []
    effective_radius = max(float(radius), 0.35)
    for p, n, b in zip(pts, normals, binormals):
        ring = p[None, :] + effective_radius * (
            cos_a[:, None] * n[None, :] + sin_a[:, None] * b[None, :]
        )
        vertices_list.append(ring)
    vertices = np.vstack(vertices_list)

    i_indices = []
    j_indices = []
    k_indices = []
    for i in range(len(pts) - 1):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            p00 = i * n_radial + j
            p01 = i * n_radial + jn
            p10 = (i + 1) * n_radial + j
            p11 = (i + 1) * n_radial + jn
            i_indices.extend([p00, p01])
            j_indices.extend([p10, p10])
            k_indices.extend([p01, p11])

    start_cap_idx = len(vertices)
    end_cap_idx = len(vertices) + 1
    vertices = np.vstack([vertices, pts[0], pts[-1]])
    for j in range(n_radial):
        jn = (j + 1) % n_radial
        i_indices.append(start_cap_idx)
        j_indices.append(jn)
        k_indices.append(j)

        i_indices.append(end_cap_idx)
        j_indices.append((len(pts) - 1) * n_radial + j)
        k_indices.append((len(pts) - 1) * n_radial + jn)

    return (
        vertices,
        np.asarray(i_indices, dtype=int),
        np.asarray(j_indices, dtype=int),
        np.asarray(k_indices, dtype=int),
    )


def create_aorta_figure(
    mask_np: np.ndarray,
    mask_image: sitk.Image,
    branches: Iterable[dict[str, Any]] = (),
    *,
    show_vessels: bool = True,
    show_centerlines: bool = True,
    show_markers: bool = True,
    show_cones: bool = False,
    **kwargs: Any,
) -> go.Figure:
    """Create an interactive 3D aorta mesh with realistic 3D vessel branch tubes."""
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
            lighting={
                "ambient": 0.4,
                "diffuse": 0.8,
                "specular": 0.2,
                "roughness": 0.5,
            },
        )
    )

    for index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{index + 1:03d}"))
        color = BRANCH_COLORS[index % len(BRANCH_COLORS)]
        radius = float(branch.get("radius_mm", 1.0))
        path_len = float(branch.get("path_length_mm", 0.0))

        ostium = np.asarray(branch.get("ostium_xyz_mm", (0, 0, 0)), dtype=float)
        seed = np.asarray(branch.get("seed_xyz_mm", (0, 0, 0)), dtype=float)
        direction = np.asarray(branch.get("direction_xyz", (0, 0, 1)), dtype=float)

        raw_path = branch.get("centreline_xyz_mm")
        if raw_path and len(raw_path) >= 2:
            path_pts = np.asarray(raw_path, dtype=float)
        else:
            path_pts = np.linspace(ostium, seed, 6)

        # 1. Realistic 3D vessel tubular mesh
        if show_vessels:
            mesh_data = _build_tube_mesh(path_pts, radius=radius)
            if mesh_data is not None:
                verts, I, J, K = mesh_data
                figure.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=I,
                        j=J,
                        k=K,
                        name=f"{branch_id} vessel",
                        color=color,
                        opacity=0.92,
                        flatshading=False,
                        lighting={
                            "ambient": 0.45,
                            "diffuse": 0.85,
                            "specular": 0.35,
                            "roughness": 0.35,
                        },
                        legendgroup=branch_id,
                        showlegend=True,
                        hoverinfo="text",
                        hovertext=(
                            f"<b>{branch_id}</b> (vessel)<br>"
                            f"Radius: {radius:.2f} mm<br>"
                            f"Trace extent: {path_len:.1f} mm"
                        ),
                    )
                )

        # 2. High-contrast 3D centerline curve
        if show_centerlines and len(path_pts) >= 2:
            figure.add_trace(
                go.Scatter3d(
                    x=path_pts[:, 0],
                    y=path_pts[:, 1],
                    z=path_pts[:, 2],
                    mode="lines",
                    line={"color": color, "width": 4},
                    name=f"{branch_id} centerline",
                    legendgroup=branch_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # 3. Ostium and 5 mm Seed landmarks
        if show_markers:
            figure.add_trace(
                go.Scatter3d(
                    x=[ostium[0]],
                    y=[ostium[1]],
                    z=[ostium[2]],
                    mode="markers",
                    marker={"size": 5, "color": "#ffd166", "symbol": "circle"},
                    name=f"{branch_id} ostium",
                    legendgroup=branch_id,
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"<b>{branch_id}</b><br>Ostium opening",
                )
            )
            figure.add_trace(
                go.Scatter3d(
                    x=[seed[0]],
                    y=[seed[1]],
                    z=[seed[2]],
                    mode="markers",
                    marker={"size": 6, "color": color, "symbol": "diamond"},
                    name=f"{branch_id} seed",
                    legendgroup=branch_id,
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=(
                        f"<b>{branch_id}</b><br>5 mm Seed landmark<br>"
                        f"radius={radius:.2f} mm"
                    ),
                )
            )

        # 4. Optional direction cone (default hidden to avoid overlap clutter)
        if show_cones:
            figure.add_trace(
                go.Cone(
                    x=[ostium[0]],
                    y=[ostium[1]],
                    z=[ostium[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    sizemode="absolute",
                    sizeref=5,
                    showscale=False,
                    colorscale=[[0, color], [1, color]],
                    name=f"{branch_id} cone",
                    legendgroup=branch_id,
                    showlegend=False,
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
