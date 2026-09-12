"""Challenge-compatible command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.io_utils import make_prediction, save_prediction
from src.pipeline import process_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect direct aortic daughter arteries from CTA."
    )
    parser.add_argument("--image", required=True, type=Path, help="CTA NIfTI file")
    parser.add_argument(
        "--aorta-mask", required=True, type=Path, help="Aorta-mask NIfTI file"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON")
    parser.add_argument(
        "--case-id",
        help="Case identifier (defaults to the CTA filename without NIfTI suffixes)",
    )
    return parser


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = process_case(args.image, args.aorta_mask)
        prediction = make_prediction(
            args.case_id or nifti_stem(args.image), result["branches"]
        )
        save_prediction(prediction, args.output)
        print(f"Wrote {len(result['branches'])} branches to {args.output}")
    except FileNotFoundError as exc:
        sys.stderr.write(f"Error (File Not Found): {exc}\n")
        sys.exit(2)
    except ValueError as exc:
        sys.stderr.write(f"Error (Validation Failed): {exc}\n")
        sys.exit(3)
    except Exception as exc:
        sys.stderr.write(f"Error (Unexpected Failure): {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
