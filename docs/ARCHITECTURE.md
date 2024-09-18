# Architecture

DefectLens is one Python service (FastAPI) that serves a React single-page app. It runs inference
in-process and executes long jobs on a background worker backed by the database. Everything runs locally,
with no paid or external services; AI text providers are optional add-ons.

```
            Browser (React SPA)                    Line PC / MES / scripts
   Dashboard · Inspect · Batch · Camera            Bearer token, Idempotency-Key
   Review · History · Model lab · Robustness                  │
   Analytics · Datasets · Jobs · Settings                     │
                   │  cookie session + CSRF header             │
                   ▼                                           ▼
 ┌─────────────────────────────── FastAPI (app/) ─────────────────────────────────┐
 │ middleware: request log (JSON), rate limit, security headers/CSP, upload limits │
 │ routers: auth · inspections · reviews · models · datasets · analytics · jobs ·  │
 │          settings/providers/cameras/users · health/metrics                      │
 │ services: inspection (decode → quality → infer → decide → persist → artifacts) │
 │           registry (model cache, activation/rollback) · batch (ZIP safety)      │
 │           ml_jobs (train/evaluate/robustness/domain-shift) · reports (PDF/CSV)  │
 │           llm (optional providers, facts-only) · retention · storage · camera   │
 │ audit: append-only, hash-chained event log      jobs: DB-backed queue + worker  │
 └──────────────┬───────────────────────────────┬──────────────────────────────────┘
                │ SQLAlchemy 2 + Alembic         │ defectml (pure ML library, no web deps)
                ▼                                ▼
   SQLite (dev) / PostgreSQL (prod)   models: PatchCore · PaDiM · PixelStats · defect-type classifier
   users, products, models, inspections,       data: manifests, VisA/folder importers, synthetic
   reviews, feedback, jobs, audit, settings    generator, validation, leakage-free splits
                                               evaluation: metrics, benchmark, degradations, robustness
   DL_VAR_DIR/                                 inference: quality gate, OOD, policy, localisation
     defectlens.db  datasets/  models/         export: ONNX (+ parity check)
     inspections/  reports/  secret.key
```

## Inspection pipeline (`app/services/inspection.py`, `defectml/inference.py`)

1. **Acquisition**: single upload, batch (multi-file or ZIP), browser webcam (interval capture), or a
   server-side camera adapter (`usb`, `ip` RTSP/HTTP stream via OpenCV, `http_snapshot`, `folder`).
2. **Safe decoding** (`defectml/imaging.py`): extension allow-list, size limit, Pillow
   decompression-bomb guard, explicit pixel limit, EXIF orientation applied, converted to RGB. ZIPs are
   flattened (no path traversal); symlinks, oversized members and suspicious compression ratios are rejected.
3. **Duplicate check**: SHA-256 of the bytes. Identical uploads for the same product/model within the
   window return the existing result (`Idempotency-Key` is also honoured).
4. **Quality gate**: resolution, brightness, contrast, sharpness (Laplacian variance) and saturated
   pixels. Any issue forces REVIEW.
5. **Inference**: the product's *active, approved* model computes an anomaly map plus an image score (the
   map maximum). An image-level embedding distance gives an **OOD score**.
6. **Decision** (`defectml/decision.py`), deliberately conservative:
   - `score ≥ fail_threshold` → **FAIL**
   - `score < pass_threshold` **and** no quality/OOD/secondary-model flags → **PASS**
   - everything else, including every error path → **REVIEW**
   - severity comes from the normalised score and the defect area.
   - optional *secondary model*: if it disagrees, the decision goes to REVIEW.
7. **Explanation**: heatmap on a fixed per-model colour scale, overlay, binary mask, and region boxes and
   polygons from connected components above the segmentation threshold. When a FAIL/REVIEW has no such
   component, the peak area is boxed and flagged `approximate`. A defect-type classifier, if trained on
   annotated defect classes, adds a type or "unknown".
8. **Persistence and audit**: inspection row, images (original, display, heatmap, overlay, mask) under
   `DL_VAR_DIR/inspections/YYYY/MM/`, and an audit event. REVIEW items enter the review queue.
9. **Human verification**: an inspector confirms or corrects the label, defect type and severity, draws
   regions, and can propose the sample for the feedback dataset.
10. **Controlled improvement**: an admin approves or rejects feedback, then builds a versioned feedback
    dataset (with provenance), trains a *candidate* model, compares it in Model lab, approves it and
    activates it. Rollback is one click. Nothing retrains or activates automatically.

## Models (`defectml/models/`)

| Model | Idea | Implementation notes |
|---|---|---|
| **PatchCore** | k-NN distance of local patch features to a memory bank of normal patches | ResNet-18 (ImageNet) layer2+layer3, 3×3 local averaging; 160 random patches per training image, then greedy k-center coreset (10 %, max 16 000) on a 128-d random projection; Gaussian smoothing σ = 4 |
| **PaDiM** | per-position multivariate Gaussian of normal features; Mahalanobis distance | ResNet-18 layer1–3 resampled to a 28×28 grid, 100 random channels, covariance regularised with 0.01·I |
| **PixelStats** | per-pixel mean/std of aligned, blurred normal images; z-score map | No deep learning and no weights download. Fast CPU baseline and fallback; requires aligned parts |
| **Defect-type classifier** | small MLP on region embeddings | Only trained when the dataset has *genuine* defect-type annotations (never derived from masks); abstains as "unknown" |

All models are implemented in-house on PyTorch/NumPy rather than via Anomalib. That keeps the dependency
footprint small, runs on a 2 GB CPU machine, allows ONNX export with a verified parity check, and makes
cancellation and progress reporting explicit. The trade-off is that ResNet-18/224 px trails the published
WideResNet-50 numbers (see `docs/MODELS.md`).

**Thresholds** come from the *validation* split only: `fail` = Youden-J optimum (balanced TPR/FPR).
`pass` = min(threshold keeping 98 % recall, fail − 25 % of the normal score range), which creates a REVIEW
band. `seg` = pixel-F1 optimum on validation masks. Admins can override thresholds per product (audited,
with a reason). Raw scores are unitless; the UI also shows `score / fail_threshold` and a Platt-calibrated
probability, clearly labelled as such.

## Background jobs (`app/jobs.py`)

Jobs (batch, train, evaluate, robustness, domain shift, retention, feedback build) are rows in the `jobs`
table. A worker thread (default `DL_WORKER_MODE=thread`), or a separate process via `defectlens worker`
(`external`), claims queued jobs atomically, sends heartbeats, reports progress, honours cancellation
between steps, and retries up to `max_attempts`. On startup, `running` jobs with a stale heartbeat are
re-queued or failed. Training is never started automatically.

## Model registry and versioning (`app/services/registry.py`)

Each trained model is a directory (`model.json`, arrays `*.npz`, `val_metrics.json`, `test_metrics.json`,
optional ONNX) registered in the DB with status `candidate → approved` (or `rejected`), later `archived`. Only approved models can
be activated, and activation/rollback history is kept per product. Loaded models are held in an LRU cache
(`DL_MODEL_CACHE_SIZE`) and warmed at startup.

## Security

See [`SECURITY.md`](../SECURITY.md): Argon2id, JWT cookie + CSRF header, token revocation, RBAC on every
route, rate limiting, strict CSP, encrypted provider keys, a hash-chained audit log, and no images sent to AI
providers.

## Frontend (`frontend/src`)

React 19 + TypeScript + Vite + Tailwind 4, TanStack Query for server state, Recharts for charts, and
lucide icons. Pages live in `pages/`; shared result rendering lives in `components/InspectionResult.tsx`
and `components/inspection.tsx` (layer tabs, opacity slider, zoom, region boxes). There is no inline script
(CSP `script-src 'self'`). The theme is initialised by `public/theme-init.js` to avoid a flash. The SPA
talks only to relative `/api` URLs, proxied by Vite in development and served by FastAPI in production.

## Configuration

All settings are environment variables prefixed `DL_` (see [`.env.example`](../.env.example)), loaded from
the repository-root `.env`. Runtime state lives under `DL_VAR_DIR` (default `<repo>/var`, gitignored).
Admin-editable settings (decision policy, quality limits, privacy, retention, providers, cameras) are stored
in the database and audited.
