"""Quick local inspection of a CTA / aorta-mask pair."""

from __future__ import annotations

import argparse

from src.io_utils import load_image
from src.visualization import create_slice_figure


parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--aorta-mask", required=True)
args = parser.parse_args()

image, image_np = load_image(args.image)
_, mask_np = load_image(args.aorta_mask)

print("CT shape:", image_np.shape)
print("Mask shape:", mask_np.shape)
print("Size:", image.GetSize())
print("Spacing:", image.GetSpacing())
print("Origin:", image.GetOrigin())
print("Direction:", image.GetDirection())
print("Intensity:", float(image_np.min()), float(image_np.max()))

create_slice_figure(image_np, mask_np > 0, image_np.shape[0] // 2).show()
