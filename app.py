"""Streamlit entry point for BranchSeed."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.io_utils import make_prediction
from src.pipeline import process_case
from src.visualization import create_aorta_figure, create_slice_figure


st.set_page_config(page_title="BranchSeed", page_icon="🫀", layout="wide")
st.title("BranchSeed")
st.caption("Automatic direct aortic daughter-artery detection from CTA")

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

if not run_clicked:
    st.info("Upload a CTA volume and matching aorta mask to begin.")
    st.stop()

with tempfile.TemporaryDirectory(prefix="branchseed-") as temp_dir:
    temp_path = Path(temp_dir)
    ct_path = temp_path / (ct_upload.name or "image.nii.gz")
    mask_path = temp_path / (mask_upload.name or "mask.nii.gz")
    ct_path.write_bytes(ct_upload.getvalue())
    mask_path.write_bytes(mask_upload.getvalue())

    try:
        with st.spinner("Processing 3D volumes..."):
            result = process_case(ct_path, mask_path)
    except Exception as exc:  # Streamlit should show actionable input errors.
        st.error(f"Processing failed: {exc}")
        st.stop()

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
    "Slice (z)", 0, result["image_np"].shape[0] - 1, result["image_np"].shape[0] // 2
)
st.pyplot(
    create_slice_figure(result["image_np"], result["mask_np"], slice_index),
    use_container_width=True,
)
