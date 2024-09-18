# Manufacturing Visual Defect Detection

Detects visual defects in manufacturing images with a local inspection pipeline and review UI.

## Features

- End-to-end inspection workflow: upload, batch, webcam or line-camera images, quality gate, inference, and PASS/FAIL/REVIEW.
- Conservative fail-safe decisions: uncertain, low-quality, or out-of-distribution images go to human review instead of auto-passing.
- PatchCore and PaDiM detectors plus a classical PixelStats baseline, with ONNX export and parity checks.
- Human review queue with verified feedback, audit trail, and versioned model retraining with approval and rollback.
- Traceable reports with heatmaps, overlays, masks, bounding boxes, confidence, and PDF/CSV export.
- Local CPU-friendly deployment with SQLite or PostgreSQL, role-based access, and optional GPU profile.

## Stack

Python, Node.js, FastAPI, Uvicorn, SQLAlchemy, Pydantic, NumPy, Pillow, SciPy, scikit-learn

## Clone

```bash
git clone https://github.com/sanjaymouraly/manufacturing-visual-defect-detection.git
cd manufacturing-visual-defect-detection
```

## Setup

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Frontend:

```bash
cd frontend
npm install
npm run build
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
uvicorn app.main:app --reload --port 8000
npm run dev                              # http://localhost:5173, proxies /api to :8000
defectml --help
defectlens --help
```

## Tests

```bash
pytest -q
```

## Notes

- Sample data, when included, is synthetic.

## License

MIT © 2024 [Sanjay Mouraly](https://github.com/sanjaymouraly)
