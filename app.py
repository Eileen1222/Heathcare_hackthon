"""Streamlit entry point for BranchSeed."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.io_utils import make_prediction
from src.pipeline import process_case
from src.visualization import (
    create_aorta_figure,
    create_slice_figure,
    prepare_aorta_3d_geometry,
)


st.set_page_config(page_title="AortiX", page_icon="🫀", layout="wide")
st.markdown(
    """
    <style>
      header[data-testid="stHeader"] {
        background: transparent;
      }
      header[data-testid="stHeader"] button,
      header[data-testid="stHeader"] svg {
        color: #e2e8f0;
        fill: #e2e8f0;
      }
      div[data-testid="stElementContainer"]:has(.aortix-topbar) {
        position: fixed;
        inset: 0 0 auto 0;
        z-index: 999;
      }
      .aortix-topbar {
        box-sizing: border-box;
        min-height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 5rem;
        text-align: center;
        border-bottom: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(10, 16, 28, 0.94);
        backdrop-filter: blur(14px);
        color: #f8fafc;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
      }
      .aortix-topbar strong {
        color: #22d3ee;
        font-size: 1.35rem;
        letter-spacing: -0.025em;
      }
      .aortix-divider {
        margin: 0 0.75rem;
        color: #64748b;
      }
      .case-overview-card {
        margin: 1.5rem 0 1rem;
        padding: 0.4rem 0;
      }
      .case-overview-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.8rem;
      }
      .case-overview-eyebrow,
      .case-overview-label {
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .case-overview-id {
        font-size: 0.95rem;
        font-weight: 650;
      }
      .case-overview-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .case-overview-stat {
        padding: 0 1rem;
        border-left: 1px solid rgba(148, 163, 184, 0.28);
      }
      .case-overview-stat:first-child {
        padding-left: 0;
        border-left: 0;
      }
      .case-overview-value {
        margin-top: 0.2rem;
        font-size: 1rem;
        font-weight: 650;
      }
      @media (max-width: 700px) {
        .aortix-topbar {
          padding-left: 3.25rem;
          font-size: 0.86rem;
        }
        .aortix-topbar strong {
          font-size: 1.1rem;
        }
        .case-overview-stats {
          grid-template-columns: 1fr;
        }
      }
    </style>
    <div class="aortix-topbar">
      <strong>AortiX</strong>
      <span class="aortix-divider">|</span>
      <span>Intelligent Aortic Branch Analysis</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def upload_signature(upload: Any) -> tuple[str, int, str | None]:
    """Return a stable identity for an uploaded file across Streamlit reruns."""
    return (
        upload.name,
        upload.size,
        getattr(upload, "file_id", None),
    )


def render_prediction_viewer(prediction_text: str) -> None:
    """Render a bounded JSON panel with a browser-native fullscreen control."""
    escaped_prediction = html.escape(prediction_text)
    viewer_html = """
    <style>
      html, body {
        margin: 0;
        height: 100%;
        background: transparent;
        font-family: "Source Sans Pro", sans-serif;
      }
      .prediction-viewer {
        box-sizing: border-box;
        height: 515px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 0.5rem;
        background: #0e1117;
        color: #fafafa;
      }
      .prediction-toolbar {
        flex: 0 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-height: 42px;
        padding: 0 0.75rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.35);
      }
      .prediction-toolbar span {
        font-size: 0.85rem;
        color: #b8c2cc;
      }
      .fullscreen-button {
        border: 1px solid rgba(128, 128, 128, 0.55);
        border-radius: 0.35rem;
        padding: 0.35rem 0.65rem;
        background: transparent;
        color: #fafafa;
        cursor: pointer;
      }
      .fullscreen-button:hover {
        border-color: #ff4b4b;
        color: #ff6b6b;
      }
      pre {
        flex: 1 1 auto;
        box-sizing: border-box;
        min-height: 0;
        margin: 0;
        padding: 0.85rem;
        overflow: auto;
        white-space: pre;
        font: 0.82rem/1.45 "Source Code Pro", monospace;
      }
      .prediction-viewer:fullscreen {
        width: 100vw;
        height: 100vh;
        border: 0;
        border-radius: 0;
      }
      .prediction-viewer:fullscreen pre {
        font-size: 1rem;
        padding: 1.5rem;
      }
    </style>
    <div id="prediction-viewer" class="prediction-viewer">
      <div class="prediction-toolbar">
        <span>prediction.json</span>
        <button id="fullscreen-button" class="fullscreen-button" type="button">
          ⛶ Full screen
        </button>
      </div>
      <pre>__PREDICTION_JSON__</pre>
    </div>
    <script>
      const viewer = document.getElementById("prediction-viewer");
      const button = document.getElementById("fullscreen-button");

      button.addEventListener("click", async () => {
        if (document.fullscreenElement) {
          await document.exitFullscreen();
        } else {
          await viewer.requestFullscreen();
        }
      });

      document.addEventListener("fullscreenchange", () => {
        button.textContent = document.fullscreenElement
          ? "× Exit full screen"
          : "⛶ Full screen";
      });
    </script>
    """.replace("__PREDICTION_JSON__", escaped_prediction)
    components.html(viewer_html, height=515, scrolling=False)


if "result" not in st.session_state:
    st.session_state.result = None
if "input_signature" not in st.session_state:
    st.session_state.input_signature = None
if "active_case_id" not in st.session_state:
    st.session_state.active_case_id = "uploaded_case"
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "prepared_3d_geometry" not in st.session_state:
    st.session_state.prepared_3d_geometry = None
# Bump when 3D mesh generation changes so stale session caches are rebuilt.
_GEOMETRY_CACHE_VERSION = 9
if st.session_state.get("geometry_cache_version") != _GEOMETRY_CACHE_VERSION:
    st.session_state.prepared_3d_geometry = None
    st.session_state.geometry_cache_version = _GEOMETRY_CACHE_VERSION

with st.sidebar:
    st.header("Case input")
    case_id = st.text_input("Case ID", value="uploaded_case")
    ct_upload = st.file_uploader("CTA volume", type=["nii", "gz"])
    mask_upload = st.file_uploader("Aorta mask", type=["nii", "gz"])
    run_clicked = st.button(
        "Run detection",
        type="primary",
        use_container_width=True,
        disabled=ct_upload is None or mask_upload is None,
    )

current_signature = None
if ct_upload is not None and mask_upload is not None:
    current_signature = (
        upload_signature(ct_upload),
        upload_signature(mask_upload),
    )

# Never display results generated from a different pair of uploaded volumes.
if current_signature != st.session_state.input_signature:
    st.session_state.result = None
    st.session_state.prediction = None
    st.session_state.prepared_3d_geometry = None
    st.session_state.input_signature = current_signature
    for state_key in (
        "slice_z",
        "slice_y",
        "slice_x",
        "2d_sl_z",
        "2d_sl_y",
        "2d_sl_x",
        "focused_branch_id",
        "active_slice_slider",
        "_ct_slider_sync",
        "_slice_external_rev",
        "slice_plane",
        "branch_selector_widget",
        "c3d_plane_type",
    ):
        st.session_state.pop(state_key, None)

if run_clicked:
    st.session_state.result = None
    st.session_state.active_case_id = case_id

    with tempfile.TemporaryDirectory(prefix="branchseed-") as temp_dir:
        temp_path = Path(temp_dir)
        ct_path = temp_path / (ct_upload.name or "image.nii.gz")
        mask_path = temp_path / (mask_upload.name or "mask.nii.gz")
        ct_path.write_bytes(ct_upload.getvalue())
        mask_path.write_bytes(mask_upload.getvalue())

        try:
            with st.spinner("Processing 3D volumes..."):
                st.session_state.result = process_case(ct_path, mask_path)
                st.session_state.prediction = make_prediction(
                    st.session_state.active_case_id,
                    st.session_state.result["branches"],
                )
                st.session_state.prepared_3d_geometry = (
                    prepare_aorta_3d_geometry(
                        st.session_state.result["mask_np"],
                        st.session_state.result["mask_image"],
                        st.session_state.result["branches"],
                    )
                )
        except Exception as exc:  # Streamlit should show actionable input errors.
            st.error(f"Processing failed: {exc}")
            st.stop()

if st.session_state.result is None:
    st.info("Upload a CTA volume and matching aorta mask to begin.")
    st.stop()

result = st.session_state.result
if st.session_state.prediction is None:
    st.session_state.prediction = make_prediction(
        st.session_state.active_case_id, result["branches"]
    )
if st.session_state.prepared_3d_geometry is None:
    st.session_state.prepared_3d_geometry = prepare_aorta_3d_geometry(
        result["mask_np"], result["mask_image"], result["branches"]
    )
prediction = st.session_state.prediction
volume_size = result["image"].GetSize()
volume_spacing = result["image"].GetSpacing()
case_id_display = html.escape(str(st.session_state.active_case_id))
st.markdown(
    f"""
    <div class="case-overview-card">
      <div class="case-overview-header">
        <div class="case-overview-eyebrow">Case overview</div>
        <div class="case-overview-id">{case_id_display}</div>
      </div>
      <div class="case-overview-stats">
        <div class="case-overview-stat">
          <div class="case-overview-label">Candidate branches</div>
          <div class="case-overview-value">{len(result["branches"])}</div>
        </div>
        <div class="case-overview-stat">
          <div class="case-overview-label">Volume dimensions</div>
          <div class="case-overview-value">
            {volume_size[0]} × {volume_size[1]} × {volume_size[2]}
          </div>
        </div>
        <div class="case-overview-stat">
          <div class="case-overview-label">Voxel spacing</div>
          <div class="case-overview-value">
            {volume_spacing[0]:.2f} × {volume_spacing[1]:.2f} ×
            {volume_spacing[2]:.2f} mm
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize interactive linkage session states
shape_z, shape_y, shape_x = (int(v) for v in result["image_np"].shape)
if "slice_z" not in st.session_state:
    st.session_state.slice_z = shape_z // 2
if "slice_y" not in st.session_state:
    st.session_state.slice_y = shape_y // 2
if "slice_x" not in st.session_state:
    st.session_state.slice_x = shape_x // 2
# Always-mounted 2D sliders use dedicated keys so plane switches never unmount
# a key (Streamlit deletes unmounted widget values — that was wiping slice_z).
if "2d_sl_z" not in st.session_state:
    st.session_state["2d_sl_z"] = int(st.session_state.slice_z)
if "2d_sl_y" not in st.session_state:
    st.session_state["2d_sl_y"] = int(st.session_state.slice_y)
if "2d_sl_x" not in st.session_state:
    st.session_state["2d_sl_x"] = int(st.session_state.slice_x)
if "focused_branch_id" not in st.session_state:
    st.session_state.focused_branch_id = None

# --- Clinical Synchronizer & Branch Inspector ---
branch_list = result["branches"]
branch_ids = [str(b["instance_id"]) for b in branch_list]
select_options = ["None (Overview)"] + branch_ids


def select_branch() -> None:
    """Update focus and linked slices before Streamlit's single rerun."""
    selected = st.session_state.branch_selector_widget
    if selected == "None (Overview)":
        st.session_state.focused_branch_id = None
        return
    st.session_state.focused_branch_id = selected
    branch = next(
        (item for item in branch_list if str(item["instance_id"]) == selected),
        None,
    )
    if branch and branch.get("ostium_zyx"):
        oz, oy, ox = [int(round(value)) for value in branch["ostium_zyx"]]
        # Write both the 3D-linkage aliases and the always-mounted 2D slider keys.
        st.session_state.slice_z = oz
        st.session_state.slice_y = oy
        st.session_state.slice_x = ox
        st.session_state["2d_sl_z"] = oz
        st.session_state["2d_sl_y"] = oy
        st.session_state["2d_sl_x"] = ox


def jump_to_branch(coordinate_key: str) -> None:
    """Move all linked 2D/3D planes before the normal widget rerun."""
    branch = next(
        (
            item
            for item in branch_list
            if str(item["instance_id"]) == st.session_state.focused_branch_id
        ),
        None,
    )
    if branch is None:
        return
    z, y, x = [int(round(value)) for value in branch[coordinate_key]]
    st.session_state.slice_z = z
    st.session_state.slice_y = y
    st.session_state.slice_x = x
    st.session_state["2d_sl_z"] = z
    st.session_state["2d_sl_y"] = y
    st.session_state["2d_sl_x"] = x


def reset_branch_focus() -> None:
    st.session_state.focused_branch_id = None
    st.session_state.branch_selector_widget = "None (Overview)"


current_sel_idx = 0
if st.session_state.focused_branch_id in branch_ids:
    current_sel_idx = branch_ids.index(st.session_state.focused_branch_id) + 1

focused_branch = next(
    (
        branch
        for branch in branch_list
        if str(branch["instance_id"]) == st.session_state.focused_branch_id
    ),
    None,
)

st.divider()
st.subheader("Clinical Branch Inspector & 3D ↔ 2D Synchronizer")
sync_c1, sync_c2, sync_c3, sync_c4 = st.columns([3, 2, 2, 2])
with sync_c1:
    st.selectbox(
        "Select branch to inspect & cross-verify:",
        options=select_options,
        index=current_sel_idx,
        key="branch_selector_widget",
        on_change=select_branch,
    )

with sync_c2:
    st.write("")
    st.write("")
    st.button(
        "Jump 2D to Ostium",
        use_container_width=True,
        disabled=focused_branch is None,
        on_click=jump_to_branch,
        args=("ostium_zyx",),
    )

with sync_c3:
    st.write("")
    st.write("")
    st.button(
        "Jump 2D to 5mm Seed",
        use_container_width=True,
        disabled=focused_branch is None,
        on_click=jump_to_branch,
        args=("seed_zyx",),
    )

with sync_c4:
    st.write("")
    st.write("")
    st.button(
        "Reset",
        use_container_width=True,
        disabled=focused_branch is None,
        on_click=reset_branch_focus,
    )

if focused_branch:
    oz, oy, ox = [int(round(v)) for v in focused_branch["ostium_zyx"]]
    sz, sy, sx = [int(round(v)) for v in focused_branch["seed_zyx"]]
    shape = result["image_np"].shape
    oz_c = min(shape[0] - 1, max(0, oz))
    oy_c = min(shape[1] - 1, max(0, oy))
    ox_c = min(shape[2] - 1, max(0, ox))
    sz_c = min(shape[0] - 1, max(0, sz))
    sy_c = min(shape[1] - 1, max(0, sy))
    sx_c = min(shape[2] - 1, max(0, sx))

    ost_hu = int(result["image_np"][oz_c, oy_c, ox_c])
    seed_hu = int(result["image_np"][sz_c, sy_c, sx_c])
    radius = float(focused_branch["radius_mm"])
    path_len = float(focused_branch.get("path_length_mm", 0.0))

    card_c1, card_c2, card_c3, card_c4, card_c5 = st.columns(5)
    card_c1.metric("Focused Branch", focused_branch["instance_id"])
    card_c2.metric("Estimated Radius", f"{radius:.2f} mm", f"Dia: {radius * 2:.2f} mm")
    card_c3.metric("Centerline Extent", f"{path_len:.1f} mm")
    card_c4.metric(
        "Ostium CT Density",
        f"{ost_hu} HU",
        "Contrast lumen" if ost_hu >= 200 else "Low contrast",
    )
    card_c5.metric(
        "5mm Seed Density",
        f"{seed_hu} HU",
        "Lumen confirmed" if seed_hu >= 180 else "Tissue boundary",
    )

st.subheader("Interactive 3D aorta & branch vessels")
c3d_1, c3d_2, c3d_3, c3d_4 = st.columns([1, 1, 1, 1])
with c3d_1:
    show_vessel_tubes = st.checkbox(
        "Vessel tubes", value=True, key="c3d_tubes"
    )
with c3d_2:
    show_centerlines_3d = st.checkbox(
        "Centerlines", value=True, key="c3d_centerlines"
    )
with c3d_3:
    show_slice_plane_3d = st.checkbox(
        "2D Slice plane", value=True, key="c3d_plane"
    )
with c3d_4:
    c3d_plane_type = st.selectbox(
        "Plane",
        ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"],
        key="c3d_plane_type",
        label_visibility="collapsed",
    )

slice_plane_info = None
if show_slice_plane_3d:
    if "Axial" in c3d_plane_type:
        slice_plane_info = ("axial", int(st.session_state.slice_z))
    elif "Coronal" in c3d_plane_type:
        slice_plane_info = ("coronal", int(st.session_state.slice_y))
    else:
        slice_plane_info = ("sagittal", int(st.session_state.slice_x))

aorta_figure = create_aorta_figure(
    result["mask_np"],
    result["mask_image"],
    result["branches"],
    show_vessels=show_vessel_tubes,
    show_centerlines=show_centerlines_3d,
    show_cones=False,
    focused_branch_id=st.session_state.focused_branch_id,
    slice_plane_info=slice_plane_info,
    prepared_geometry=st.session_state.prepared_3d_geometry,
)
aorta_figure.update_layout(height=580)
st.plotly_chart(
    aorta_figure,
    use_container_width=True,
    config={"displayModeBar": True, "scrollZoom": True},
)


def render_ct_viewer(case_result: dict[str, Any], focused_branch_id: str | None) -> None:
    """Interactive 2D CT viewer with per-axis slice state that survives plane switches."""
    st.subheader("CT / detection overlay")
    slice_plot_config = {
        "scrollZoom": True,
        "displaylogo": False,
    }
    spacing_zyx = case_result["image"].GetSpacing()[::-1]
    shape = case_result["image_np"].shape
    plane_specs = {
        "Axial (Z)": ("axial", 0, "z", "2d_sl_z"),
        "Coronal (Y)": ("coronal", 1, "y", "2d_sl_y"),
        "Sagittal (X)": ("sagittal", 2, "x", "2d_sl_x"),
    }

    def format_intersections(branches: list[dict], axis: int, slice_val: int, spacing: float) -> list[str]:
        tol = max(1.0, 0.75 * spacing)
        hits = []
        for b in branches:
            bid = str(b.get("instance_id", ""))
            ost = b["ostium_zyx"][axis]
            seed = b["seed_zyx"][axis]
            d_ost = abs(ost - slice_val) * spacing
            d_seed = abs(seed - slice_val) * spacing
            parts = []
            if d_ost <= tol:
                parts.append(f"Ostium Δ{d_ost:.1f}mm")
            if d_seed <= tol:
                parts.append(f"5mm Seed Δ{d_seed:.1f}mm")
            if parts:
                hits.append(f"**{bid}** ({', '.join(parts)})")
        return hits

    selected_view = st.radio(
        "Viewing plane",
        list(plane_specs),
        horizontal=True,
        label_visibility="collapsed",
        key="slice_plane",
    )
    plane, axis, axis_name, _active_key = plane_specs[selected_view]

    # Mount all three sliders every run. Unmounting the inactive plane's slider
    # used to delete its session value, so Z→Y→Z jumped to a default/wrong slice.
    z_col, y_col, x_col = st.columns(3)
    with z_col:
        z_idx = st.slider(
            "Axial (Z)",
            0,
            int(shape[0] - 1),
            key="2d_sl_z",
            disabled=selected_view != "Axial (Z)",
        )
    with y_col:
        y_idx = st.slider(
            "Coronal (Y)",
            0,
            int(shape[1] - 1),
            key="2d_sl_y",
            disabled=selected_view != "Coronal (Y)",
        )
    with x_col:
        x_idx = st.slider(
            "Sagittal (X)",
            0,
            int(shape[2] - 1),
            key="2d_sl_x",
            disabled=selected_view != "Sagittal (X)",
        )

    # Keep 3D linkage aliases in sync with the always-mounted 2D sliders.
    st.session_state.slice_z = int(z_idx)
    st.session_state.slice_y = int(y_idx)
    st.session_state.slice_x = int(x_idx)

    slice_index = {"axial": int(z_idx), "coronal": int(y_idx), "sagittal": int(x_idx)}[
        plane
    ]

    hits = format_intersections(
        case_result["branches"],
        axis,
        slice_index,
        spacing_zyx[axis],
    )
    if hits:
        st.info(
            f"🎯 **Current {selected_view} slice ({axis_name}={slice_index}) intersects:** "
            + " · ".join(hits)
        )

    st.plotly_chart(
        create_slice_figure(
            case_result["image_np"],
            case_result["mask_np"],
            slice_index,
            plane=plane,
            branches=case_result["branches"],
            spacing_zyx=spacing_zyx,
            focused_branch_id=focused_branch_id,
            figure_height=500,
        ),
        use_container_width=True,
        config=slice_plot_config,
        # Include plane+slice so Streamlit cannot restore a stale Plotly figure
        # from another index under a shared chart identity.
        key=f"ct_2d_{plane}_{slice_index}",
        theme=None,
    )


render_ct_viewer(result, st.session_state.focused_branch_id)

st.divider()
with st.expander("Prediction JSON", expanded=False):
    prediction_text = json.dumps(prediction, indent=2)
    render_prediction_viewer(prediction_text)
    st.download_button(
        "Download prediction.json",
        data=prediction_text.encode("utf-8"),
        file_name=f"{st.session_state.active_case_id}.json",
        mime="application/json",
        use_container_width=True,
    )
