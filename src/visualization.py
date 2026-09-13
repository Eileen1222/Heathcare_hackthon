"""Interactive 2D and 3D Plotly visualizations.

2D CT overlays use per-branch saturated colors for direction arrows and lighter
dotted same-family shades for centerlines. 3D keeps a unified clinical lumen
material; branch identity there comes from selection highlight and labels.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

# Headless / Streamlit Cloud: VTK must not require a local X display.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow")

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from skimage.measure import marching_cubes
from scipy import ndimage as ndi

from src.io_utils import physical_xyz_to_zyx

try:
    import pyvista as pv

    _HAS_PYVISTA = True
except ImportError:  # pragma: no cover - exercised only without pyvista
    pv = None
    _HAS_PYVISTA = False
except Exception:  # pragma: no cover - missing libGL etc. on bare Linux
    pv = None
    _HAS_PYVISTA = False

# Dual-layer vessel look for 3D: lighter translucent lumen + darker rim/edge.
CLINICAL_VESSEL = "#ff0000"
CLINICAL_VESSEL_INNER = "#ff7a7a"
CLINICAL_VESSEL_RIM = "#9a0000"
CLINICAL_AORTA = CLINICAL_VESSEL
CLINICAL_VESSEL_SELECTED = CLINICAL_VESSEL
CLINICAL_CENTERLINE = "#6b7280"
CLINICAL_CENTERLINE_SELECTED = "#f8fafc"
CLINICAL_OSTIUM = "#ffe066"
CLINICAL_SEED = "#7dd3d8"
CLINICAL_DIRECTION = "#ffffff"
CLINICAL_MASK_2D = "#ff7f0e"
CLINICAL_LABEL = "#ffffff"
CLINICAL_LABEL_SELECTED = "#ffe566"
SLICE_PLANE_COLORS = {
    "axial": "#94a3b8",
    "coronal": "#86a39a",
    "sagittal": "#a89f8c",
}

# 2D overlays: saturated direction colour + lighter same-family centerline.
BRANCH_PALETTE_2D: tuple[tuple[str, str], ...] = (
    ("#00a8cc", "#7ee8fa"),
    ("#2eae67", "#9cf2c0"),
    ("#e2a400", "#ffe49a"),
    ("#d94f87", "#ffadd0"),
    ("#805ad5", "#c4b5fd"),
    ("#008f72", "#7de2cb"),
)
BRANCH_COLORS: tuple[str, ...] = tuple(color for color, _ in BRANCH_PALETTE_2D)
OSTIUM_2D = "#ffe066"


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
    focused_branch_id: str | None = None,
    figure_height: int = 500,
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

    # Always emit overlays. Layer show/hide is done via Plotly legend
    # (groupclick) so Streamlit does not rebuild the chart and wipe zoom.
    # (Streamlit 1.63 identities plotly charts by full figure JSON.)
    view_revision = f"{plane}:{int(slice_index)}"
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
            uid=f"ct:{view_revision}",
            showlegend=False,
            hovertemplate=(
                "horizontal=%{x:.1f} mm<br>vertical=%{y:.1f} mm"
                "<br>HU=%{z:.0f}<extra>CT</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Heatmap(
            x=x_coordinates,
            y=y_coordinates,
            z=np.where(mask_slice, 1.0, np.nan),
            colorscale=[[0.0, CLINICAL_MASK_2D], [1.0, CLINICAL_MASK_2D]],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            opacity=0.35,
            name="Aorta mask",
            uid=f"mask:{view_revision}",
            legendgroup="layer_mask",
            showlegend=True,
            visible=True if show_mask else "legendonly",
            hoverinfo="skip",
        )
    )

    tolerance_mm = max(1.0, 0.75 * spacing[fixed_axis])
    centerline_legend_shown = False
    ostium_legend_shown = False
    direction_legend_shown = False
    seed_legend_shown = False
    radius_legend_shown = False

    for branch_index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{branch_index + 1:03d}"))
        color, centerline_color = BRANCH_PALETTE_2D[
            branch_index % len(BRANCH_PALETTE_2D)
        ]
        is_focused = focused_branch_id is not None and branch_id == focused_branch_id
        is_dimmed = focused_branch_id is not None and not is_focused
        trace_opacity = 1.0 if not is_dimmed else 0.30

        if branch.get("centreline_zyx"):
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
                show_cl_legend = not centerline_legend_shown
                centerline_legend_shown = True
                figure.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines+markers",
                        line={
                            "color": centerline_color,
                            "width": 3 if is_focused else 2,
                            "dash": "dot",
                        },
                        marker={"color": centerline_color, "size": 3},
                        opacity=trace_opacity,
                        name="Centerlines",
                        uid=f"{branch_id}:centerline:{view_revision}",
                        legendgroup="layer_centerlines",
                        showlegend=show_cl_legend,
                        visible=True if show_centerlines else "legendonly",
                        hovertemplate=f"{branch_id} centerline<extra></extra>",
                    )
                )

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
            show_ost_legend = not ostium_legend_shown
            ostium_legend_shown = True
            figure.add_trace(
                go.Scatter(
                    x=[ostium_x],
                    y=[ostium_y],
                    mode="markers+text",
                    marker={
                        "size": 12 if is_focused else 10,
                        "color": OSTIUM_2D,
                        "line": {
                            "color": color,
                            "width": 2 if is_focused else 1,
                        },
                    },
                    text=[f"★ {branch_id}" if is_focused else branch_id],
                    textposition="top center",
                    textfont={
                        "color": color,
                        "size": 11 if is_focused else 10,
                    },
                    opacity=trace_opacity,
                    name="Ostia",
                    uid=f"{branch_id}:ostium:{view_revision}",
                    legendgroup="layer_ostia",
                    showlegend=show_ost_legend,
                    visible=True if show_ostia else "legendonly",
                    hovertemplate=(
                        f"{branch_id}<br>ostium"
                        f"<br>z/y/x={ostium[0]:.1f}/{ostium[1]:.1f}/{ostium[2]:.1f}"
                        "<extra></extra>"
                    ),
                )
            )

            show_dir_legend = not direction_legend_shown
            direction_legend_shown = True
            figure.add_trace(
                go.Scatter(
                    x=[ostium_x, seed_x],
                    y=[ostium_y, seed_y],
                    mode="lines+markers",
                    line={
                        "color": color,
                        "width": 3 if is_focused else 2,
                    },
                    marker={
                        "color": color,
                        "size": [0, 12 if is_focused else 10],
                        "symbol": ["circle", "arrow"],
                        "angleref": "previous",
                    },
                    opacity=trace_opacity,
                    name="Directions",
                    uid=f"{branch_id}:direction:{view_revision}",
                    legendgroup="layer_directions",
                    showlegend=show_dir_legend,
                    visible=True if show_directions else "legendonly",
                    hovertemplate=f"{branch_id} direction<extra></extra>",
                )
            )

        if seed_near or ostium_near:
            is_projection = not seed_near
            # Projection seeds track the direction overlay; on-slice seeds are
            # independent.
            seed_on = bool(show_seeds) and (
                seed_near or bool(show_directions)
            )
            show_seed_legend = not seed_legend_shown
            seed_legend_shown = True
            figure.add_trace(
                go.Scatter(
                    x=[seed_x],
                    y=[seed_y],
                    mode="markers",
                    marker={
                        "size": 10 if is_focused else 8,
                        "symbol": "x" if is_projection else "diamond",
                        "color": color,
                        "line": {
                            "color": "#0f172a",
                            "width": 1 if is_focused else 0,
                        },
                    },
                    opacity=trace_opacity,
                    name="5 mm seeds",
                    uid=f"{branch_id}:seed:{view_revision}",
                    legendgroup="layer_seeds",
                    showlegend=show_seed_legend,
                    visible=True if seed_on else "legendonly",
                    hovertemplate=(
                        f"{branch_id}<br>"
                        + ("seed projection" if is_projection else "5 mm seed")
                        + f"<br>radius={float(branch['radius_mm']):.2f} mm"
                        + "<extra></extra>"
                    ),
                )
            )

        if seed_near:
            radius = float(branch["radius_mm"])
            angles = np.linspace(0.0, 2.0 * np.pi, 65)
            show_rad_legend = not radius_legend_shown
            radius_legend_shown = True
            figure.add_trace(
                go.Scatter(
                    x=seed_x + radius * np.cos(angles),
                    y=seed_y + radius * np.sin(angles),
                    mode="lines",
                    line={
                        "color": centerline_color,
                        "width": 2 if is_focused else 1,
                        "dash": "dot",
                    },
                    opacity=trace_opacity,
                    name="Seed radius",
                    uid=f"{branch_id}:radius:{view_revision}",
                    legendgroup="layer_radius",
                    showlegend=show_rad_legend,
                    visible=True if show_radius else "legendonly",
                    hovertemplate=(
                        f"{branch_id}<br>radius={radius:.2f} mm<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        height=figure_height,
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        title={
            "text": f"{plane.capitalize()} slice {axis_name}={slice_index}",
            "x": 0.02,
            "y": 0.98,
            "xanchor": "left",
            "font": {"size": 13},
        },
        dragmode="zoom",
        # Keep zoom/pan when Streamlit remounts with the same plane/slice
        # (e.g. focus change). Layer toggles use the legend and do not remount.
        uirevision=view_revision,
        xaxis={
            "visible": False,
            "constrain": "domain",
            "range": [float(x_coordinates[0]), float(x_coordinates[-1])],
            "uirevision": view_revision,
        },
        yaxis={
            "visible": False,
            "range": [float(y_coordinates[-1]), float(y_coordinates[0])],
            "scaleanchor": "x",
            "scaleratio": 1,
            "uirevision": view_revision,
        },
        showlegend=True,
        legend={
            "title": {"text": "Layers"},
            "groupclick": "togglegroup",
            "itemsizing": "constant",
            "tracegroupgap": 2,
            "orientation": "v",
            "yanchor": "top",
            "y": 0.98,
            "xanchor": "left",
            "x": 1.01,
            "bgcolor": "rgba(15, 23, 42, 0.88)",
            "bordercolor": "rgba(148, 163, 184, 0.35)",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#e2e8f0"},
        },
    )
    return figure


def _path_arc_length_mm(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _dense_hover_samples(
    path_pts: np.ndarray,
    *,
    step_mm: float = 1.0,
    extra_tips: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """Sample a centerline densely so Scatter3d hover stays hittable in 3D."""
    pts = np.asarray(path_pts, dtype=float)
    samples: list[np.ndarray] = []
    if len(pts) >= 2:
        length = _path_arc_length_mm(pts)
        n = max(4, int(np.ceil(max(length, step_mm) / max(step_mm, 0.25))) + 1)
        n = min(n, 64)
        if len(pts) >= 3:
            samples_arr = _resample_or_smooth_path(pts, num_points=n)
            samples.extend(np.asarray(samples_arr, dtype=float))
        else:
            t = np.linspace(0.0, 1.0, n)
            samples.extend(pts[0][None, :] * (1.0 - t[:, None]) + pts[-1][None, :] * t[:, None])
    elif len(pts) == 1:
        samples.append(pts[0])
    if extra_tips is not None:
        for tip in extra_tips:
            samples.append(np.asarray(tip, dtype=float))
    if not samples:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(samples, dtype=float)


def _branch_hover_html(
    branch: dict[str, Any],
    *,
    selected: bool = False,
) -> str:
    branch_id = str(branch.get("instance_id", "branch"))
    radius = float(branch.get("radius_mm", 1.0))
    path_len = float(branch.get("path_length_mm", 0.0))
    return (
        f"<b>{branch_id}</b>"
        + (" <b>[SELECTED]</b>" if selected else "")
        + f"<br>Radius: {radius:.2f} mm"
        + f"<br>Trace extent: {path_len:.1f} mm"
        + "<br>x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm"
        + "<extra></extra>"
    )


def _mesh_vertex_hover_text(
    vertices_xyz_mm: np.ndarray,
    branches: Sequence[dict[str, Any]],
    *,
    max_dist_mm: float = 7.0,
    focused_branch_id: str | None = None,
) -> list[str]:
    """Map each mesh vertex to the nearest branch (Plotly Mesh3d can show this on hover).

    Scatter probes inside a Mesh3d never receive hover in Plotly gl3d — the mesh
    always wins picking. Binding branch text onto the outer mesh vertices is the
    reliable workaround.
    """
    verts = np.asarray(vertices_xyz_mm, dtype=float)
    if len(verts) == 0:
        return []
    if not branches:
        return ["<b>Aorta</b><br>Vessel tree"] * len(verts)

    anchor_pts: list[np.ndarray] = []
    anchor_text: list[str] = []
    for index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{index + 1:03d}"))
        ostium = np.asarray(branch.get("ostium_xyz_mm", (0, 0, 0)), dtype=float)
        seed = np.asarray(branch.get("seed_xyz_mm", (0, 0, 0)), dtype=float)
        raw_path = branch.get("centreline_xyz_mm")
        if raw_path and len(raw_path) >= 2:
            path_pts = np.asarray(raw_path, dtype=float)
        else:
            path_pts = np.linspace(ostium, seed, 6)
        samples = _dense_hover_samples(
            path_pts, step_mm=1.2, extra_tips=(ostium, seed)
        )
        selected = focused_branch_id is not None and branch_id == focused_branch_id
        # Static HTML for mesh vertices (no %{x} placeholders — Mesh3d fills coords).
        radius = float(branch.get("radius_mm", 1.0))
        path_len = float(branch.get("path_length_mm", 0.0))
        label = (
            f"<b>{branch_id}</b>"
            + (" <b>[SELECTED]</b>" if selected else "")
            + f"<br>Radius: {radius:.2f} mm"
            + f"<br>Trace extent: {path_len:.1f} mm"
        )
        for sample in samples:
            anchor_pts.append(np.asarray(sample, dtype=float))
            anchor_text.append(label)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(anchor_pts, dtype=float))
    dists, idxs = tree.query(verts, k=1, workers=-1)
    hover_text: list[str] = []
    for dist, idx in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
        if float(dist) <= max_dist_mm:
            hover_text.append(anchor_text[int(idx)])
        else:
            hover_text.append("<b>Aorta</b><br>Vessel tree")
    return hover_text


def _adaptive_tube_sample_count(arc_length_mm: float, *, max_points: int = 48) -> int:
    """Space rings by arc length so short branches don't look like stacked washers."""
    # ~0.65 mm between cross-sections keeps walls smooth without packing rings.
    return int(np.clip(round(float(arc_length_mm) / 0.65) + 1, 6, max_points))


def _ensure_min_path_length(
    pts: np.ndarray,
    radius_mm: float,
    *,
    min_length_mm: float | None = None,
) -> np.ndarray:
    """Extend the distal tip so short ostium→seed segments read as real pipes."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts
    length = _path_arc_length_mm(pts)
    target = (
        float(min_length_mm)
        if min_length_mm is not None
        else max(9.0, 7.0 * float(radius_mm))
    )
    if length >= target - 1e-6:
        return pts
    tangent = pts[-1] - pts[-2]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        return pts
    tangent /= norm
    need = target - length
    n_extra = max(2, int(np.ceil(need / 0.6)))
    extras = np.asarray(
        [pts[-1] + tangent * (need * (i + 1) / n_extra) for i in range(n_extra)],
        dtype=float,
    )
    return np.vstack([pts, extras])


def _resample_or_smooth_path(points: np.ndarray, num_points: int = 48) -> np.ndarray:
    """Spline-smooth and resample a centerline to suppress voxel-step jagginess."""
    if len(points) <= 2:
        return np.asarray(points, dtype=float)
    diffs = np.diff(points, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    valid_idx = np.ones(len(points), dtype=bool)
    valid_idx[1:] = step_dists > 1e-4
    pts = np.asarray(points, dtype=float)[valid_idx]
    if len(pts) <= 2:
        return pts

    diffs = np.diff(pts, axis=0)
    step_dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(step_dists), 0, 0.0)
    total_len = float(cum_dist[-1])
    if total_len < 1e-4:
        return pts[:2]

    # Honour the requested count (adaptive callers pass a short-path-friendly size).
    target_count = int(np.clip(int(num_points), 6, 64))
    target_dists = np.linspace(0.0, total_len, target_count)

    # Prefer a mild smoothing spline so tube walls are not faceted by voxel noise.
    if len(pts) >= 4:
        try:
            from scipy.interpolate import splprep, splev

            u = cum_dist / total_len
            # s scales with path length / point count: enough to kill stair-steps.
            smooth = max(0.15, 0.08 * total_len)
            tck, _ = splprep(
                [pts[:, 0], pts[:, 1], pts[:, 2]],
                u=u,
                s=smooth,
                k=min(3, len(pts) - 1),
            )
            xu, yu, zu = splev(np.linspace(0.0, 1.0, target_count), tck)
            return np.column_stack([xu, yu, zu])
        except Exception:
            pass

    resampled = np.zeros((len(target_dists), 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_dists, cum_dist, pts[:, dim])
    return resampled


def _radius_profile(
    n_points: int,
    seed_radius_mm: float,
    ostium_radius_mm: float | None,
) -> np.ndarray:
    """Nearly constant lumen radius with a gentle cosine taper (avoids bulging)."""
    base = max(float(seed_radius_mm), 0.35)
    if ostium_radius_mm is not None and np.isfinite(ostium_radius_mm):
        proximal = float(np.clip(ostium_radius_mm, base * 0.95, base * 1.25))
    else:
        proximal = base * 1.12
    distal = base * 0.92
    t = np.linspace(0.0, 1.0, n_points)
    # Smooth hermite-like blend; no sharp knee that creates surface ridges.
    w = t * t * (3.0 - 2.0 * t)
    return proximal * (1.0 - w) + distal * w


def _extend_path_into_aorta(pts: np.ndarray, radius_mm: float) -> np.ndarray:
    """Push the proximal tip slightly into the aorta so the tube reads as a branch root."""
    if len(pts) < 2:
        return pts
    tangent = pts[1] - pts[0]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        return pts
    tangent /= norm
    extend_mm = max(0.9 * float(radius_mm), 0.8)
    return np.vstack([pts[0] - tangent * extend_mm, pts])


def _polyline_from_points(points: np.ndarray):
    """Build a single continuous VTK polyline cell (required for seamless tubes)."""
    assert pv is not None
    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=float)
    cell = np.arange(0, len(points), dtype=np.int_)
    cell = np.insert(cell, 0, len(points))
    poly.lines = cell
    return poly


def _faces_to_ijk(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert VTK/PyVista face array to Plotly Mesh3d i/j/k indices."""
    faces = np.asarray(faces, dtype=int).ravel()
    i_indices: list[int] = []
    j_indices: list[int] = []
    k_indices: list[int] = []
    cursor = 0
    while cursor < len(faces):
        n = int(faces[cursor])
        ids = faces[cursor + 1 : cursor + 1 + n]
        cursor += n + 1
        if n == 3:
            i_indices.append(int(ids[0]))
            j_indices.append(int(ids[1]))
            k_indices.append(int(ids[2]))
        elif n > 3:
            for offset in range(1, n - 1):
                i_indices.append(int(ids[0]))
                j_indices.append(int(ids[offset]))
                k_indices.append(int(ids[offset + 1]))
    return (
        np.asarray(i_indices, dtype=int),
        np.asarray(j_indices, dtype=int),
        np.asarray(k_indices, dtype=int),
    )


def _smooth_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    n_iter: int = 40,
    relaxation: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Laplacian-smooth a triangle surface when PyVista is available."""
    if not _HAS_PYVISTA or len(vertices) < 4 or len(faces) < 1:
        return vertices, faces
    try:
        faces_vtk = np.hstack(
            [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()
        mesh = pv.PolyData(np.asarray(vertices, dtype=float), faces_vtk)
        mesh = mesh.smooth(
            n_iter=n_iter,
            relaxation_factor=relaxation,
            feature_smoothing=False,
            boundary_smoothing=True,
        )
        mesh = mesh.triangulate()
        new_faces = mesh.faces.reshape(-1, 4)[:, 1:]
        return np.asarray(mesh.points, dtype=float), np.asarray(new_faces, dtype=int)
    except Exception:
        return vertices, faces


def _offset_mesh_along_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    offset_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Push/pull a surface along point normals (for rim vs lumen shells)."""
    if abs(float(offset_mm)) < 1e-6:
        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
    if _HAS_PYVISTA and len(vertices) >= 4 and len(faces) >= 1:
        try:
            faces_vtk = np.hstack(
                [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
            ).ravel()
            mesh = pv.PolyData(np.asarray(vertices, dtype=float), faces_vtk)
            mesh = mesh.compute_normals(
                cell_normals=False,
                point_normals=True,
                consistent_normals=True,
                auto_orient_normals=True,
                inplace=False,
            )
            normals = np.asarray(mesh.point_data["Normals"], dtype=float)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            normals = normals / norms
            offset_verts = np.asarray(mesh.points, dtype=float) + normals * float(
                offset_mm
            )
            offset_faces = mesh.faces.reshape(-1, 4)[:, 1:]
            return offset_verts, np.asarray(offset_faces, dtype=int)
        except Exception:
            pass

    # Fallback without PyVista: crude area-weighted face normals → vertex normals.
    verts = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=int)
    normals = np.zeros_like(verts)
    v0 = verts[faces_arr[:, 0]]
    v1 = verts[faces_arr[:, 1]]
    v2 = verts[faces_arr[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    for axis in range(3):
        np.add.at(normals[:, axis], faces_arr[:, 0], face_normals[:, axis])
        np.add.at(normals[:, axis], faces_arr[:, 1], face_normals[:, axis])
        np.add.at(normals[:, axis], faces_arr[:, 2], face_normals[:, axis])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals = normals / norms
    return verts + normals * float(offset_mm), faces_arr


def _volume_shading_layers(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build translucent inner lumen + darker outer rim shells."""
    faces = np.asarray(faces, dtype=int)
    vertices = np.asarray(vertices, dtype=float)
    inner_verts, inner_faces = _offset_mesh_along_normals(vertices, faces, -0.18)
    rim_verts, rim_faces = _offset_mesh_along_normals(vertices, faces, 0.32)
    return {
        "inner_vertices_xyz_mm": inner_verts,
        "inner_faces": inner_faces,
        "rim_vertices_xyz_mm": rim_verts,
        "rim_faces": rim_faces,
    }


def _prepare_tube_centerline(
    points: np.ndarray,
    radius: float,
    *,
    max_samples: int = 48,
) -> np.ndarray | None:
    """Extend short branch paths and resample by arc length (avoids stacked rings)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return None
    pts = _extend_path_into_aorta(pts, radius)
    pts = _ensure_min_path_length(pts, radius)
    n_samples = _adaptive_tube_sample_count(
        _path_arc_length_mm(pts), max_points=max_samples
    )
    pts = _resample_or_smooth_path(pts, num_points=n_samples)
    return pts if len(pts) >= 2 else None


def _build_tube_mesh_pyvista(
    points: np.ndarray,
    radius: float,
    *,
    ostium_radius_mm: float | None = None,
    n_sides: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Open pipe via PyVista: single outer wall, uncapped ends (no stacked inner rings)."""
    assert pv is not None
    pts = _prepare_tube_centerline(np.asarray(points, dtype=float), radius)
    if pts is None:
        return None
    outer_radii = _radius_profile(len(pts), radius, ostium_radius_mm)

    line = _polyline_from_points(pts)
    outer = line.copy()
    outer["radius"] = outer_radii
    n_sides = max(24, int(n_sides))
    tube = outer.tube(
        scalars="radius",
        absolute=True,
        n_sides=n_sides,
        capping=False,
    ).triangulate()
    if tube.n_points < 8 or tube.n_cells < 1:
        return None
    try:
        tube = tube.smooth(
            n_iter=20,
            relaxation_factor=0.08,
            feature_smoothing=False,
            boundary_smoothing=False,
        )
    except Exception:
        pass
    i_idx, j_idx, k_idx = _faces_to_ijk(tube.faces)
    if i_idx.size == 0:
        return None
    return np.asarray(tube.points, dtype=float), i_idx, j_idx, k_idx


def _build_tube_mesh_bishop(
    points: np.ndarray,
    radius: float,
    *,
    ostium_radius_mm: float | None = None,
    n_radial: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Open pipe: smooth proximal blend into aorta; distal end keeps an open-tube lip."""
    pts = _prepare_tube_centerline(np.asarray(points, dtype=float), radius)
    if pts is None:
        return None
    outer_radii = _radius_profile(len(pts), radius, ostium_radius_mm)
    # Distal wall thickness only — proximal (aorta side) must not look like a pipe mouth.
    lip_radii = np.maximum(outer_radii * 0.72, outer_radii - 0.45)
    lip_radii = np.minimum(lip_radii, outer_radii * 0.88)

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
            ni = ni / max(np.linalg.norm(ni), 1e-8)
        else:
            axis_unit = axis / axis_len
            cos_theta = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = float(np.arccos(cos_theta))
            v = normals[-1]
            ni = (
                v * np.cos(theta)
                + np.cross(axis_unit, v) * np.sin(theta)
                + axis_unit * np.dot(axis_unit, v) * (1.0 - np.cos(theta))
            )
            ni = ni / max(np.linalg.norm(ni), 1e-8)
        normals.append(ni)
        binormals.append(np.cross(t_curr, ni))

    # Drop the most proximal rings that sit at the aorta wall so the fused surface
    # owns the junction; keep the distal open-pipe tip intact.
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    )
    total_len = float(arc[-1]) if len(arc) else 0.0
    proximal_cut_mm = min(max(1.6, 1.8 * float(radius)), 0.28 * max(total_len, 1e-3))
    keep = arc >= (proximal_cut_mm - 1e-6)
    # Always keep enough samples for a real distal tube + open mouth.
    if int(np.count_nonzero(keep)) < 4:
        keep = np.ones(len(pts), dtype=bool)
        keep[: max(0, len(pts) - 4)] = False

    pts = pts[keep]
    outer_radii = outer_radii[keep]
    lip_radii = lip_radii[keep]
    normals = [normals[i] for i, flag in enumerate(keep) if flag]
    binormals = [binormals[i] for i, flag in enumerate(keep) if flag]

    angles = np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    outer_rings = []
    for p, n, b, r_out in zip(pts, normals, binormals, outer_radii):
        outer_rings.append(
            p[None, :]
            + float(r_out)
            * (cos_a[:, None] * n[None, :] + sin_a[:, None] * b[None, :])
        )
    outer_verts = np.vstack(outer_rings)
    n_pts = len(pts)
    last = n_pts - 1

    # Distal open-port lip only (outer end = uncapped pipe). No proximal lip.
    lip_distal = pts[last][None, :] + float(lip_radii[last]) * (
        cos_a[:, None] * normals[last][None, :]
        + sin_a[:, None] * binormals[last][None, :]
    )
    vertices = np.vstack([outer_verts, lip_distal])
    lip_distal_base = len(outer_verts)

    i_indices: list[int] = []
    j_indices: list[int] = []
    k_indices: list[int] = []

    def _quad(a: int, b: int, c: int, d: int) -> None:
        i_indices.extend([a, a])
        j_indices.extend([b, c])
        k_indices.extend([c, d])

    for i in range(n_pts - 1):
        for j in range(n_radial):
            jn = (j + 1) % n_radial
            o00 = i * n_radial + j
            o01 = i * n_radial + jn
            o10 = (i + 1) * n_radial + j
            o11 = (i + 1) * n_radial + jn
            _quad(o00, o10, o11, o01)

    # Distal annular mouth only — lumen stays open, pipe look preserved at outer tip.
    for j in range(n_radial):
        jn = (j + 1) % n_radial
        o0 = last * n_radial + j
        o1 = last * n_radial + jn
        _quad(o0, o1, lip_distal_base + jn, lip_distal_base + j)

    return (
        vertices,
        np.asarray(i_indices, dtype=int),
        np.asarray(j_indices, dtype=int),
        np.asarray(k_indices, dtype=int),
    )


def _build_tube_mesh(
    points: Sequence[Sequence[float]],
    radius: float,
    *,
    ostium_radius_mm: float | None = None,
    n_radial: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Branch tube: smooth aorta-side junction; distal tip stays an open pipe."""
    raw_pts = np.asarray(points, dtype=float)
    if len(raw_pts) < 2:
        return None

    # Prefer Bishop builder: proximal blend + distal annular open mouth.
    mesh = _build_tube_mesh_bishop(
        raw_pts,
        radius,
        ostium_radius_mm=ostium_radius_mm,
        n_radial=max(16, min(int(n_radial), 28)),
    )
    if mesh is not None:
        return mesh

    if _HAS_PYVISTA:
        try:
            return _build_tube_mesh_pyvista(
                raw_pts,
                radius,
                ostium_radius_mm=ostium_radius_mm,
                n_sides=n_radial,
            )
        except Exception:
            return None
    return None


def _stamp_sphere(
    volume: np.ndarray,
    center_zyx: Sequence[float],
    radius_zyx: Sequence[float],
) -> None:
    """OR an anisotropic sphere into a boolean volume (in-place)."""
    cz, cy, cx = (float(v) for v in center_zyx)
    rz, ry, rx = (max(float(v), 0.55) for v in radius_zyx)
    shape = volume.shape
    z0 = max(0, int(np.floor(cz - rz - 1)))
    z1 = min(shape[0], int(np.ceil(cz + rz + 2)))
    y0 = max(0, int(np.floor(cy - ry - 1)))
    y1 = min(shape[1], int(np.ceil(cy + ry + 2)))
    x0 = max(0, int(np.floor(cx - rx - 1)))
    x1 = min(shape[2], int(np.ceil(cx + rx + 2)))
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    sphere = (
        ((zz - cz) / rz) ** 2
        + ((yy - cy) / ry) ** 2
        + ((xx - cx) / rx) ** 2
    ) <= 1.0
    volume[z0:z1, y0:y1, x0:x1] |= sphere


def _fuse_vessel_tree_surface(
    aorta_mask: np.ndarray,
    mask_image: sitk.Image,
    branches: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp branch tubes into the aorta mask and extract one smooth surface.

    Voxel union + morphological closing + Gaussian isosurface yields organic
    junctions similar to a reference vessel-tree model, without brittle mesh CSG.
    """
    binary = np.asarray(aorta_mask, dtype=bool)
    branches = list(branches)
    if not branches:
        verts_zyx, faces, _, _ = marching_cubes(binary.astype(np.uint8), level=0.5)
        verts_xyz = _indices_to_physical(verts_zyx, mask_image)
        return _smooth_triangle_mesh(verts_xyz, faces, n_iter=55, relaxation=0.14)

    # Work in a padded crop for speed on large CTA volumes.
    occupied = [np.flatnonzero(binary.any(axis=tuple(i for i in range(3) if i != a)))
                for a in range(3)]
    pad = 28
    z0 = max(0, int(occupied[0][0]) - pad)
    z1 = min(binary.shape[0], int(occupied[0][-1]) + pad + 1)
    y0 = max(0, int(occupied[1][0]) - pad)
    y1 = min(binary.shape[1], int(occupied[1][-1]) + pad + 1)
    x0 = max(0, int(occupied[2][0]) - pad)
    x1 = min(binary.shape[2], int(occupied[2][-1]) + pad + 1)
    crop = binary[z0:z1, y0:y1, x0:x1].copy()
    offset = np.array([z0, y0, x0], dtype=float)
    spacing_xyz = np.asarray(mask_image.GetSpacing(), dtype=float)

    for branch in branches:
        ostium = np.asarray(branch.get("ostium_xyz_mm", (0, 0, 0)), dtype=float)
        seed = np.asarray(branch.get("seed_xyz_mm", (0, 0, 0)), dtype=float)
        raw_path = branch.get("centreline_xyz_mm")
        if raw_path and len(raw_path) >= 2:
            path_pts = np.asarray(raw_path, dtype=float)
        else:
            path_pts = np.linspace(ostium, seed, 8)
        radius = float(branch.get("radius_mm", 1.0))
        ostium_r = branch.get("ostium_radius_mm")
        ostium_r = float(ostium_r) if ostium_r is not None else None
        path_pts = _prepare_tube_centerline(path_pts, radius, max_samples=40)
        if path_pts is None:
            continue
        radii = _radius_profile(len(path_pts), radius, ostium_r)
        # Inflate enough that sub-voxel / thin CTA branches survive smoothing.
        radii = np.maximum(radii * 1.25, max(radius, 0.85))

        for point_xyz, radius_mm in zip(path_pts, radii):
            zyx = np.asarray(physical_xyz_to_zyx(mask_image, point_xyz), dtype=float)
            local = zyx - offset
            # Keep at least ~1.4 voxels so thin daughters are not erased.
            rx = max(float(radius_mm) / max(spacing_xyz[0], 1e-6), 1.4)
            ry = max(float(radius_mm) / max(spacing_xyz[1], 1e-6), 1.4)
            rz = max(float(radius_mm) / max(spacing_xyz[2], 1e-6), 1.4)
            _stamp_sphere(crop, local, (rz, ry, rx))

        # Mild ostium-only fillet (inner junction). Do not alter distal branch tips.
        fillet_r = max(float(radii[0]) * 1.35, radius * 1.8)
        direction = path_pts[min(1, len(path_pts) - 1)] - path_pts[0]
        dnorm = float(np.linalg.norm(direction))
        direction = direction / dnorm if dnorm > 1e-8 else np.array([0.0, 0.0, 1.0])
        for step_mm, scale in ((0.0, 1.0), (1.0, 0.85)):
            point_xyz = ostium - direction * step_mm
            zyx = np.asarray(physical_xyz_to_zyx(mask_image, point_xyz), dtype=float)
            local = zyx - offset
            r_mm = fillet_r * scale
            rx = max(r_mm / max(spacing_xyz[0], 1e-6), 1.4)
            ry = max(r_mm / max(spacing_xyz[1], 1e-6), 1.4)
            rz = max(r_mm / max(spacing_xyz[2], 1e-6), 1.4)
            _stamp_sphere(crop, local, (rz, ry, rx))

    # Close ostium gaps without eroding thin distal branches away.
    struct = ndi.generate_binary_structure(3, 1)
    crop = ndi.binary_closing(crop, structure=struct, iterations=2)

    volume = ndi.gaussian_filter(crop.astype(np.float32), sigma=0.8)
    verts_local, faces, _, _ = marching_cubes(volume, level=0.38)
    verts_zyx = verts_local + offset
    verts_xyz = _indices_to_physical(verts_zyx, mask_image)
    return _smooth_triangle_mesh(verts_xyz, faces, n_iter=50, relaxation=0.13)


def prepare_aorta_3d_geometry(
    mask_np: np.ndarray,
    mask_image: sitk.Image,
    branches: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Precompute fused vessel-tree surface + optional per-branch tubes."""
    binary = np.asarray(mask_np) > 0
    if not np.any(binary):
        raise ValueError("Cannot visualize an empty aorta mask")

    branches = list(branches)
    tree_vertices, tree_faces = _fuse_vessel_tree_surface(binary, mask_image, branches)
    shading = _volume_shading_layers(tree_vertices, tree_faces)

    # Keep an aorta-only surface for the "hide daughters" toggle.
    verts_zyx, aorta_faces, _, _ = marching_cubes(binary.astype(np.uint8), level=0.5)
    aorta_vertices = _indices_to_physical(verts_zyx, mask_image)
    aorta_vertices, aorta_faces = _smooth_triangle_mesh(
        aorta_vertices, aorta_faces, n_iter=40, relaxation=0.12
    )
    aorta_shading = _volume_shading_layers(aorta_vertices, aorta_faces)

    bounds = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        occupied = np.flatnonzero(binary.any(axis=other_axes))
        bounds.append((int(occupied[0]), int(occupied[-1])))

    branch_meshes = {}
    for index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{index + 1:03d}"))
        ostium = np.asarray(branch.get("ostium_xyz_mm", (0, 0, 0)), dtype=float)
        seed = np.asarray(branch.get("seed_xyz_mm", (0, 0, 0)), dtype=float)
        raw_path = branch.get("centreline_xyz_mm")
        if raw_path and len(raw_path) >= 2:
            path_pts = np.asarray(raw_path, dtype=float)
        else:
            path_pts = np.linspace(ostium, seed, 6)
        # Floor display radius so sub-millimetre CTA twigs stay visible in 3D.
        display_radius = max(float(branch.get("radius_mm", 1.0)), 0.6)
        branch_meshes[branch_id] = _build_tube_mesh(
            path_pts,
            radius=display_radius,
            ostium_radius_mm=(
                float(branch["ostium_radius_mm"])
                if branch.get("ostium_radius_mm") is not None
                else None
            ),
        )

    return {
        "shape_zyx": tuple(binary.shape),
        "bounds_zyx": tuple(bounds),
        "aorta_vertices_xyz_mm": aorta_vertices,
        "aorta_faces": aorta_faces,
        "aorta_inner_vertices_xyz_mm": aorta_shading["inner_vertices_xyz_mm"],
        "aorta_inner_faces": aorta_shading["inner_faces"],
        "aorta_rim_vertices_xyz_mm": aorta_shading["rim_vertices_xyz_mm"],
        "aorta_rim_faces": aorta_shading["rim_faces"],
        "tree_vertices_xyz_mm": tree_vertices,
        "tree_faces": tree_faces,
        "tree_inner_vertices_xyz_mm": shading["inner_vertices_xyz_mm"],
        "tree_inner_faces": shading["inner_faces"],
        "tree_rim_vertices_xyz_mm": shading["rim_vertices_xyz_mm"],
        "tree_rim_faces": shading["rim_faces"],
        "branch_meshes": branch_meshes,
    }


def create_aorta_figure(
    mask_np: np.ndarray,
    mask_image: sitk.Image,
    branches: Iterable[dict[str, Any]] = (),
    *,
    show_vessels: bool = True,
    show_centerlines: bool = True,
    show_markers: bool = True,
    show_cones: bool = False,
    focused_branch_id: str | None = None,
    slice_plane_info: tuple[str, int] | None = None,
    prepared_geometry: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create an interactive 3D fused vessel tree (aorta + daughters, one surface)."""
    branches = list(branches)
    geometry = prepared_geometry or prepare_aorta_3d_geometry(
        mask_np, mask_image, branches
    )
    use_tree = show_vessels and geometry.get("tree_vertices_xyz_mm") is not None
    if use_tree:
        body_vertices = geometry.get(
            "tree_inner_vertices_xyz_mm", geometry["tree_vertices_xyz_mm"]
        )
        body_faces = geometry.get("tree_inner_faces", geometry["tree_faces"])
        rim_vertices = geometry.get(
            "tree_rim_vertices_xyz_mm", geometry["tree_vertices_xyz_mm"]
        )
        rim_faces = geometry.get("tree_rim_faces", geometry["tree_faces"])
        body_name = "Vessel lumen"
        rim_name = "Vessel rim"
    else:
        body_vertices = geometry.get(
            "aorta_inner_vertices_xyz_mm", geometry["aorta_vertices_xyz_mm"]
        )
        body_faces = geometry.get("aorta_inner_faces", geometry["aorta_faces"])
        rim_vertices = geometry.get(
            "aorta_rim_vertices_xyz_mm", geometry["aorta_vertices_xyz_mm"]
        )
        rim_faces = geometry.get("aorta_rim_faces", geometry["aorta_faces"])
        body_name = "Aorta lumen"
        rim_name = "Aorta rim"

    volume_lighting = {
        "ambient": 0.16,
        "diffuse": 0.95,
        "specular": 0.62,
        "roughness": 0.28,
        "fresnel": 0.35,
    }
    light_pos = {"x": 90, "y": 140, "z": 220}

    # Bind branch hover text to the OUTER rim mesh. Plotly gl3d always picks Mesh3d
    # over interior Scatter3d, so this is the only reliable hover path.
    rim_hover_text = _mesh_vertex_hover_text(
        rim_vertices,
        branches,
        max_dist_mm=7.5,
        focused_branch_id=focused_branch_id,
    )

    # Outer rim first (darker edge), then translucent lighter lumen on top.
    figure = go.Figure(
        data=[
            go.Mesh3d(
                x=rim_vertices[:, 0],
                y=rim_vertices[:, 1],
                z=rim_vertices[:, 2],
                i=rim_faces[:, 0],
                j=rim_faces[:, 1],
                k=rim_faces[:, 2],
                name=rim_name,
                color=CLINICAL_VESSEL_RIM,
                opacity=0.78,
                flatshading=False,
                lighting=volume_lighting,
                lightposition=light_pos,
                legendgroup="tissue",
                showlegend=True,
                text=rim_hover_text,
                hovertext=rim_hover_text,
                hovertemplate=(
                    "%{hovertext}"
                    "<br>x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm"
                    "<extra></extra>"
                ),
                hoverinfo="text",
            ),
            go.Mesh3d(
                x=body_vertices[:, 0],
                y=body_vertices[:, 1],
                z=body_vertices[:, 2],
                i=body_faces[:, 0],
                j=body_faces[:, 1],
                k=body_faces[:, 2],
                name=body_name,
                color=CLINICAL_VESSEL_INNER,
                opacity=0.48,
                flatshading=False,
                lighting={
                    "ambient": 0.34,
                    "diffuse": 0.88,
                    "specular": 0.45,
                    "roughness": 0.4,
                    "fresnel": 0.12,
                },
                lightposition=light_pos,
                legendgroup="tissue",
                showlegend=True,
                # Inner shell stays non-interactive so rim hover wins cleanly.
                hoverinfo="none",
            ),
        ]
    )

    ostium_legend_shown = False
    seed_legend_shown = False
    labels_legend_shown = False

    for index, branch in enumerate(branches):
        branch_id = str(branch.get("instance_id", f"branch_{index + 1:03d}"))
        radius = float(branch.get("radius_mm", 1.0))
        path_len = float(branch.get("path_length_mm", 0.0))

        is_focused = focused_branch_id is not None and branch_id == focused_branch_id
        is_dimmed = focused_branch_id is not None and not is_focused

        centerline_w = 4 if is_focused else (1 if is_dimmed else 2)
        ostium_s = 7 if is_focused else (3 if is_dimmed else 4)
        seed_s = 7 if is_focused else (3 if is_dimmed else 5)

        ostium = np.asarray(branch.get("ostium_xyz_mm", (0, 0, 0)), dtype=float)
        seed = np.asarray(branch.get("seed_xyz_mm", (0, 0, 0)), dtype=float)
        direction = np.asarray(branch.get("direction_xyz", (0, 0, 1)), dtype=float)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-8:
            direction = direction / direction_norm
        else:
            direction = np.array([0.0, 0.0, 1.0])

        raw_path = branch.get("centreline_xyz_mm")
        if raw_path and len(raw_path) >= 2:
            path_pts = np.asarray(raw_path, dtype=float)
        else:
            path_pts = np.linspace(ostium, seed, 6)

        branch_hover = _branch_hover_html(branch, selected=is_focused)

        if show_vessels:
            mesh_data = geometry.get("branch_meshes", {}).get(branch_id)
            # Daughter tubes: keep hovertemplate on the protruding mesh surface.
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
                        name=(
                            "Selected branch"
                            if is_focused
                            else f"{branch_id} vessel"
                        ),
                        color=CLINICAL_VESSEL if is_focused else CLINICAL_VESSEL_INNER,
                        opacity=(
                            0.95
                            if is_focused
                            else (0.12 if is_dimmed else 0.88)
                        ),
                        flatshading=False,
                        lighting={
                            "ambient": 0.4 if is_focused else 0.32,
                            "diffuse": 0.9,
                            "specular": 0.55 if is_focused else 0.4,
                            "roughness": 0.25,
                        },
                        legendgroup=branch_id,
                        showlegend=bool(is_focused),
                        hovertemplate=branch_hover,
                    )
                )

        if show_centerlines and len(path_pts) >= 2:
            cl_color = (
                CLINICAL_CENTERLINE_SELECTED if is_focused else CLINICAL_CENTERLINE
            )
            figure.add_trace(
                go.Scatter3d(
                    x=path_pts[:, 0],
                    y=path_pts[:, 1],
                    z=path_pts[:, 2],
                    mode="lines",
                    line={"color": cl_color, "width": centerline_w},
                    opacity=0.9 if is_focused else (0.1 if is_dimmed else 0.22),
                    name=f"{branch_id} centerline",
                    showlegend=False,
                    hovertemplate=branch_hover,
                )
            )

        if show_vessels and (not is_dimmed or is_focused):
            label_offset = direction * max(radius * 2.4, 2.8)
            label_pos = ostium + label_offset
            figure.add_trace(
                go.Scatter3d(
                    x=[label_pos[0]],
                    y=[label_pos[1]],
                    z=[label_pos[2]],
                    mode="text",
                    text=[branch_id],
                    textposition="middle right",
                    textfont={
                        "size": 14 if is_focused else 11,
                        "color": (
                            CLINICAL_LABEL_SELECTED if is_focused else CLINICAL_LABEL
                        ),
                        "family": "Segoe UI, Helvetica, Arial, sans-serif",
                    },
                    opacity=1.0 if not is_dimmed else 0.35,
                    name="Branch labels",
                    legendgroup="labels",
                    showlegend=not labels_legend_shown,
                    hovertemplate=branch_hover,
                )
            )
            labels_legend_shown = True

        if show_markers:
            figure.add_trace(
                go.Scatter3d(
                    x=[ostium[0]],
                    y=[ostium[1]],
                    z=[ostium[2]],
                    mode="markers",
                    marker={
                        "size": ostium_s,
                        "color": CLINICAL_OSTIUM,
                        "symbol": "circle",
                        "line": {
                            "color": CLINICAL_VESSEL,
                            "width": 2 if is_focused else 1,
                        },
                    },
                    opacity=1.0 if not is_dimmed else 0.2,
                    name="Ostium",
                    legendgroup="markers",
                    showlegend=not ostium_legend_shown,
                    hovertemplate=branch_hover,
                )
            )
            ostium_legend_shown = True
            figure.add_trace(
                go.Scatter3d(
                    x=[seed[0]],
                    y=[seed[1]],
                    z=[seed[2]],
                    mode="markers",
                    marker={
                        "size": seed_s,
                        "color": CLINICAL_SEED,
                        "symbol": "diamond",
                        "line": {
                            "color": "#ffffff" if is_focused else CLINICAL_SEED,
                            "width": 2 if is_focused else 0,
                        },
                    },
                    opacity=1.0 if not is_dimmed else 0.2,
                    name="5 mm seed",
                    legendgroup="markers",
                    showlegend=not seed_legend_shown,
                    hovertemplate=branch_hover,
                )
            )
            seed_legend_shown = True

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
                    sizeref=6 if is_focused else 4,
                    showscale=False,
                    colorscale=[
                        [0, CLINICAL_DIRECTION],
                        [1, CLINICAL_DIRECTION],
                    ],
                    name=f"{branch_id} cone",
                    showlegend=False,
                )
            )

    if slice_plane_info is not None:
        plane_type, slice_index = slice_plane_info
        plane_type = plane_type.lower()
        shape_z, shape_y, shape_x = geometry["shape_zyx"]
        (z_bound, y_bound, x_bound) = geometry["bounds_zyx"]
        pad = 18
        z_min = max(0, z_bound[0] - pad)
        z_max = min(shape_z - 1, z_bound[1] + pad)
        y_min = max(0, y_bound[0] - pad)
        y_max = min(shape_y - 1, y_bound[1] + pad)
        x_min = max(0, x_bound[0] - pad)
        x_max = min(shape_x - 1, x_bound[1] + pad)

        corners_zyx = None
        plane_color = SLICE_PLANE_COLORS.get(plane_type, "#94a3b8")
        plane_name = f"{plane_type.capitalize()} plane"
        if plane_type == "axial":
            cur_z = max(0, min(shape_z - 1, int(slice_index)))
            corners_zyx = np.array([
                [cur_z, y_min, x_min],
                [cur_z, y_min, x_max],
                [cur_z, y_max, x_max],
                [cur_z, y_max, x_min],
            ], dtype=float)
            plane_name = f"Axial plane (z={cur_z})"
        elif plane_type == "coronal":
            cur_y = max(0, min(shape_y - 1, int(slice_index)))
            corners_zyx = np.array([
                [z_min, cur_y, x_min],
                [z_min, cur_y, x_max],
                [z_max, cur_y, x_max],
                [z_max, cur_y, x_min],
            ], dtype=float)
            plane_name = f"Coronal plane (y={cur_y})"
        elif plane_type == "sagittal":
            cur_x = max(0, min(shape_x - 1, int(slice_index)))
            corners_zyx = np.array([
                [z_min, y_min, cur_x],
                [z_max, y_min, cur_x],
                [z_max, y_max, cur_x],
                [z_min, y_max, cur_x],
            ], dtype=float)
            plane_name = f"Sagittal plane (x={cur_x})"

        if corners_zyx is not None:
            corners_xyz = _indices_to_physical(corners_zyx, mask_image)
            figure.add_trace(
                go.Mesh3d(
                    x=corners_xyz[:, 0],
                    y=corners_xyz[:, 1],
                    z=corners_xyz[:, 2],
                    i=[0, 0],
                    j=[1, 2],
                    k=[2, 3],
                    color=plane_color,
                    opacity=0.16,
                    name=plane_name,
                    showlegend=True,
                    hoverinfo="none",
                )
            )
            closed_perimeter = np.vstack([corners_xyz, corners_xyz[0]])
            figure.add_trace(
                go.Scatter3d(
                    x=closed_perimeter[:, 0],
                    y=closed_perimeter[:, 1],
                    z=closed_perimeter[:, 2],
                    mode="lines",
                    line={"color": plane_color, "width": 3},
                    name=f"{plane_name} outline",
                    showlegend=False,
                    hoverinfo="none",
                )
            )

    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"color": "#f5f5f5"},
        hovermode="closest",
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (mm)",
            "yaxis_title": "y (mm)",
            "zaxis_title": "z (mm)",
            "xaxis": {
                "backgroundcolor": "#000000",
                "gridcolor": "#222222",
                "showbackground": True,
            },
            "yaxis": {
                "backgroundcolor": "#000000",
                "gridcolor": "#222222",
                "showbackground": True,
            },
            "zaxis": {
                "backgroundcolor": "#000000",
                "gridcolor": "#222222",
                "showbackground": True,
            },
            "bgcolor": "#000000",
            "hovermode": "closest",
        },
        legend={
            "orientation": "h",
            "bgcolor": "rgba(0,0,0,0.55)",
            "bordercolor": "#333333",
            "font": {"size": 11},
        },
    )
    return figure


def _indices_to_physical(vertices_zyx: np.ndarray, image: sitk.Image) -> np.ndarray:
    indices_xyz = vertices_zyx[:, ::-1]
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return (direction @ (indices_xyz * spacing).T).T + origin
