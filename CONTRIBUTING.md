# Contributing to DefectLens

Thanks for helping! This project inspects physical products, and a wrong PASS can ship a defective part.
Please keep that in mind: **conservative behaviour, honest metrics and tests come before features.**

## Ground rules

1. **No fabricated numbers.** Every metric in docs, UI or PRs must come from a reproducible command (see
   `docs/MODELS.md`). Mark synthetic-data results as synthetic.
2. **Never tune on the held-out test split.** Thresholds and calibration use the validation split only.
   Test-split numbers are for reporting.
3. **Fail safe.** Anything that prevents a proper inspection (decode error, missing model, bad image
   quality, out-of-distribution input) must route to REVIEW, never PASS.
4. **Datasets and weights stay out of git.** No images beyond `samples/`, no `*.npz`/`*.onnx`/`*.pt`, no
   secrets. Respect dataset licenses (see `NOTICE`, `docs/DATASETS.md`); MVTec AD is non-commercial.
5. **No silent model changes.** Training produces *candidate* versions. Only an admin can approve and
   activate a model, and every change is audited.

## Development setup

See the README *Native install* section. In short:

```bash
cd backend && python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[export,dev]"
cd ../frontend && npm ci
```

Run the API with `uvicorn app.main:app --reload --port 8000` (from `backend/`) and the UI with
`npm run dev` (from `frontend/`, proxied to the API at http://localhost:5173).

## Checks (CI runs all of these)

```bash
# backend (from backend/)
ruff check .
mypy app defectml
pytest                          # fast suite (~25 s), offline, no pretrained weights needed
pytest -m slow                  # trains PatchCore/PaDiM + ONNX parity (downloads ResNet-18 weights once)
python ../scripts/check_symbols.py
python ../scripts/check_api_routes.py   # every /api URL used by the frontend exists in the OpenAPI schema

# frontend (from frontend/)
npm run typecheck
npm run build
# end-to-end (API must be running with a built frontend, see docs/TESTING.md)
npx playwright test
```

## Pull requests

- Keep PRs focused, and describe *why* as well as *what*.
- Add or adjust tests: unit tests for `defectml/`, API tests in `backend/tests/app/`, Playwright specs for
  UI flows.
- Database changes need an Alembic migration (`alembic revision -m "..."`). Prefer native `op.add_column`
  over `batch_alter_table` on populated tables; see the comment in `0002_password_change_required.py`.
  `tests/app/test_batch_models_ops.py` checks that the models and migrations match and that upgrades work
  on a populated database.
- UI changes: keep every control functional and accessible (labels, keyboard, contrast, dark mode). If the
  change is visible in the docs, regenerate screenshots with `npm run screenshots`.
- Changes to the decision policy, thresholds or metrics need a note in `docs/MODELS.md`.

## Commit style

Conventional-ish prefixes help: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`.

By contributing you agree that your contributions are licensed under Apache-2.0.
