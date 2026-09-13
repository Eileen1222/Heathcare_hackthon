# Aortix

AortiX is a CPU-friendly prototype for detecting direct daughter arteries
from a CTA volume and an aorta mask. The repository has one shared Python
pipeline and two entry points:

- `run.py` writes the challenge-compatible JSON prediction.
- `app.py` runs the same pipeline in a Streamlit interface with 2D and 3D
  visual checks.

The branch detector is intentionally left as a clearly marked baseline seam so
the team can develop detection and geometry independently without changing the
CLI or web interface.

## Setup (Windows PowerShell)

Python 3.11 or 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the CLI

```powershell
python run.py `
  --image data\subject001\orig1.nii.gz `
  --aorta-mask data\subject001\mask1.nii.gz `
  --output outputs\predictions\subject001.json `
  --case-id subject001
```

## Run the web app

```powershell
python -m streamlit run app.py
```

Upload a CT and its aorta mask. The app calls `src.pipeline.process_case`,
shows a mask overlay and interactive 3D aorta, and exposes the same prediction
JSON used by the CLI.

## Test

```powershell
python -m unittest discover -s tests -v
```

## Coordinate convention

- NumPy array coordinates are always `(z, y, x)` and use the suffix `_zyx`.
- SimpleITK physical coordinates are always `(x, y, z)` millimetres and use
  the suffix `_xyz_mm`.

Never write voxel indices into `ostium_xyz_mm` or `seed_xyz_mm`.


## Data layout

```text
data/
  subject001/
    orig1.nii.gz
    mask1.nii.gz
outputs/
  predictions/
  figures/
```

Medical data and generated outputs are ignored by Git.
