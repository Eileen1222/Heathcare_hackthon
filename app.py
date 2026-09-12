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
from src.visualization import create_aorta_figure, create_slice_figure


st.set_page_config(page_title="BranchSeed", page_icon="🫀", layout="wide")
st.title("BranchSeed")
st.caption("Automatic direct aortic daughter-artery detection from CTA")


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
    st.session_state.input_signature = current_signature
    for slice_key in ("slice_z", "slice_y", "slice_x"):
        st.session_state.pop(slice_key, None)

if run_clicked:
    # Clear the previous result before processing so a failed rerun cannot show
    # stale data for the current input pair.
    st.session_state.result = None

    with tempfile.TemporaryDirectory(prefix="branchseed-") as temp_dir:
        temp_path = Path(temp_dir)
        ct_path = temp_path / (ct_upload.name or "image.nii.gz")
        mask_path = temp_path / (mask_upload.name or "mask.nii.gz")
        ct_path.write_bytes(ct_upload.getvalue())
        mask_path.write_bytes(mask_upload.getvalue())

        try:
            with st.spinner("Processing 3D volumes..."):
                st.session_state.result = process_case(ct_path, mask_path)
        except Exception as exc:  # Streamlit should show actionable input errors.
            st.error(f"Processing failed: {exc}")
            st.stop()

if st.session_state.result is None:
    st.info("Upload a CTA volume and matching aorta mask to begin.")
    st.stop()

result = st.session_state.result
prediction = make_prediction(case_id, result["branches"])
st.success(f'Detected {len(result["branches"])} candidate branches')

viewer_col, details_col = st.columns([2, 1])
with viewer_col:
    st.subheader("Interactive 3D aorta & branch vessels")
    c3d_1, c3d_2, c3d_3 = st.columns(3)
    with c3d_1:
        show_vessel_tubes = st.checkbox(
            "Vessel tubes (0-10 mm)", value=True, key="c3d_tubes"
        )
    with c3d_2:
        show_centerlines_3d = st.checkbox(
            "3D Centerlines", value=True, key="c3d_centerlines"
        )
    with c3d_3:
        show_cones_3d = st.checkbox(
            "Direction cones", value=False, key="c3d_cones"
        )

    aorta_figure = create_aorta_figure(
        result["mask_np"],
        result["mask_image"],
        result["branches"],
        show_vessels=show_vessel_tubes,
        show_centerlines=show_centerlines_3d,
        show_cones=show_cones_3d,
    )
    aorta_figure.update_layout(height=600)
    st.plotly_chart(
        aorta_figure,
        use_container_width=True,
    )

with details_col:
    st.subheader("Prediction")
    prediction_text = json.dumps(prediction, indent=2)
    render_prediction_viewer(prediction_text)
    payload = prediction_text.encode("utf-8")
    st.download_button(
        "Download prediction.json",
        data=payload,
        file_name=f"{case_id}.json",
        mime="application/json",
        use_container_width=True,
    )

st.subheader("CT / detection overlay")
slice_plot_config = {
    "scrollZoom": True,
    "displaylogo": False,
}
spacing_zyx = result["image"].GetSpacing()[::-1]
slice_view_col, layer_controls_col = st.columns([5, 1], gap="medium")

with layer_controls_col:
    st.markdown("#### Layers")
    show_mask = st.checkbox("🟧 Aorta mask", value=True)
    show_ostia = st.checkbox("🟡 Ostia", value=True)
    show_seeds = st.checkbox("◆ 5 mm seeds", value=True)
    show_directions = st.checkbox("➜ Directions", value=True)
    show_centerlines = st.checkbox("┈ Centerlines", value=True)
    show_radius = st.checkbox("◯ Seed radius", value=True)
    st.caption(
        "Each branch has one saturated color. Its centerline uses a lighter "
        "dotted shade of the same color. An × marks a seed projected from a "
        "nearby slice."
    )

slice_options = {
    "branches": result["branches"],
    "spacing_zyx": spacing_zyx,
    "show_mask": show_mask,
    "show_ostia": show_ostia,
    "show_seeds": show_seeds,
    "show_directions": show_directions,
    "show_centerlines": show_centerlines,
    "show_radius": show_radius,
}

with slice_view_col:
    axial_tab, coronal_tab, sagittal_tab = st.tabs(
        ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]
    )
    with axial_tab:
        slice_z = st.slider(
            "Axial slice (z)",
            0,
            result["image_np"].shape[0] - 1,
            result["image_np"].shape[0] // 2,
            key="slice_z",
        )
        st.plotly_chart(
            create_slice_figure(
                result["image_np"],
                result["mask_np"],
                slice_z,
                plane="axial",
                **slice_options,
            ),
            use_container_width=True,
            config=slice_plot_config,
        )

    with coronal_tab:
        slice_y = st.slider(
            "Coronal slice (y)",
            0,
            result["image_np"].shape[1] - 1,
            result["image_np"].shape[1] // 2,
            key="slice_y",
        )
        st.plotly_chart(
            create_slice_figure(
                result["image_np"],
                result["mask_np"],
                slice_y,
                plane="coronal",
                **slice_options,
            ),
            use_container_width=True,
            config=slice_plot_config,
        )

    with sagittal_tab:
        slice_x = st.slider(
            "Sagittal slice (x)",
            0,
            result["image_np"].shape[2] - 1,
            result["image_np"].shape[2] // 2,
            key="slice_x",
        )
        st.plotly_chart(
            create_slice_figure(
                result["image_np"],
                result["mask_np"],
                slice_x,
                plane="sagittal",
                **slice_options,
            ),
            use_container_width=True,
            config=slice_plot_config,
        )
