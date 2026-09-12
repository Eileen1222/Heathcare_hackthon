"""Streamlit entry point for BranchSeed."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

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
    st.session_state.pop("slice_z", None)

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
    st.subheader("Interactive 3D aorta")
    st.plotly_chart(
        create_aorta_figure(
            result["mask_np"], result["mask_image"], result["branches"]
        ),
        use_container_width=True,
    )

with details_col:
    st.subheader("Prediction")
    st.json(prediction)
    payload = json.dumps(prediction, indent=2).encode("utf-8")
    st.download_button(
        "Download prediction.json",
        data=payload,
        file_name=f"{case_id}.json",
        mime="application/json",
        use_container_width=True,
    )

st.subheader("Axial CT / mask overlay")
slice_index = st.slider(
    "Slice (z)",
    0,
    result["image_np"].shape[0] - 1,
    result["image_np"].shape[0] // 2,
    key="slice_z",
)
st.pyplot(
    create_slice_figure(result["image_np"], result["mask_np"], slice_index),
    use_container_width=True,
)
