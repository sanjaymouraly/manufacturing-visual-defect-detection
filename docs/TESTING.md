# Testing

| Suite | Command (see below for setup) | Size | Runtime* | Needs |
|---|---|---|---|---|
| Backend fast | `pytest` (from `backend/`) | 105 tests | ≈ 15–25 s | nothing: offline, no pretrained weights, temp DB |
| Backend slow | `pytest -m slow -o addopts=""` | 2 tests | ≈ 20 s (plus a one-time weights download) | ResNet-18 weights (downloaded once into `TORCH_HOME`) |
| Static checks | `ruff check . ../scripts` · `mypy app defectml` | 65 source files | ≈ 20 s | — |
| Consistency | `python ../scripts/check_symbols.py` · `check_api_routes.py` · `gen_api_docs.py --check` | — | seconds | — |
| Frontend | `npm run typecheck` · `npm run build` | — | ≈ 25 s | Node 20 |
| End-to-end | `npx playwright test` (from `frontend/`) | 18 tests (+1 screenshot capture) | ≈ 40 s | running API with built UI and demo models |
| Docs screenshots | `npm run screenshots` | 17 PNGs | ≈ 40 s | as E2E |

\* measured on a 2-vCPU / 2 GB development VM.

## What is covered

**ML core (`tests/test_defectml_core.py`, 58 tests)**
- *Safe decoding*: RGB/grayscale, garbage, truncated files, pixel bombs rejected before decoding, tiny
  images, disallowed formats, stable hashes, quality flags (dark, blurry).
- *Decision policy*: FAIL at or above the reject threshold; PASS only below the pass threshold with no
  flags; borderline → REVIEW. Inspection-impossible cases never PASS (parametrised), NaN score → REVIEW,
  quality and OOD flags block auto-PASS, secondary-model disagreement, policy switches, critical FAILs
  queued for confirmation, threshold validation.
- *Metrics*: AUROC matches scikit-learn, bootstrap CI contains the point estimate, Wilson interval,
  recall-target threshold, Youden prevalence-independence (unlike F1), the threshold ordering that creates
  the review band, monotone and bounded Platt calibration.
- *Splits and data*: train is normal-only, groups never straddle splits, deterministic per seed, exact
  duplicates across splits detected.
- *Other*: degradation contract for all 19 degradations (shape/dtype preserved, deterministic per seed,
  severity 5 not milder than 1 except orientation/background change, unknown names rejected), region
  coordinates in original-image space, greedy coreset equals naive k-center, approximate peak box for
  FAILs without a segmented region, CLI storage-dir resolution.

**API and services (`tests/app/`, 47 tests; real FastAPI app, temporary SQLite DB, small trained models)**
- *Auth and security (17)*: public vs protected routes; wrong password rejected and audited; cookie
  `httpOnly`/`SameSite=Strict`; CSRF header required for cookie mutations; bearer flow; logout and password
  change revoke tokens; weak passwords rejected; RBAC for viewer/inspector; login rate limit; audit chain
  valid, append-only, and detects raw-SQL tampering; security headers and CSP (`frame-ancestors` in
  production); password-change-required lifecycle; seeded demo accounts flagged.
- *Inspection workflow (13)*: active models, normal vs anomalous decisions, artifacts and PDF report,
  duplicate/idempotency handling, corrupt uploads, corrupt or blank camera frames fail safe to REVIEW,
  413 oversize, unknown product, review → feedback → approval, review-region validation, history
  filters and sort, dashboard counts consistent with the DB.
- *Batch, models, ops (17)*: ZIP bomb, file-count limit, path traversal, symlinks, non-images and bad ZIPs;
  mixed ZIP + image batch; job cancel; retention dry run changes nothing; only approved models activate,
  plus rollback; threshold override needs a reason, is audited and changes decisions; model card metrics
  come from real files; evaluation job; provider keys never returned; AI-free summary labelled as a
  template; Alembic fresh upgrade/downgrade/autogenerate check; **migrations on a populated DB with
  foreign keys enforced**.

**Slow (`tests/test_deep_models_slow.py`)**: trains PatchCore and PaDiM on synthetic data, evaluates,
runs inference, exports ONNX and checks PyTorch/ONNX Runtime parity.

**End-to-end (`frontend/e2e/`, Chromium, against the real backend)**
- `auth.spec`: form login/logout and server-side session revocation; a new account is prompted to replace
  its admin-set password and changes it in the UI; admin vs viewer navigation; the server enforces roles
  and CSRF independently of the UI; theme toggle persists.
- `inspect.spec`: a defective sample is not auto-passed and shows heatmap/overlay/mask layers; re-uploading
  the same image returns the existing result unless forced; history and detail page show the audit trail.
- `workflow.spec`: background batch with results; inspector reviews a queued item with a drawn region and
  proposes it as feedback; admin rejects pending feedback in Datasets.
- `admin.spec`: model lab lists versions and the model detail shows held-out metrics; robustness lab
  previews a degradation with clean vs degraded decisions; analytics and the AI box work with no LLM
  configured; the policy settings section saves; providers can be added, tested (no network needed) and
  deleted; the jobs page lists background jobs.

## Running the end-to-end tests

```bash
# terminal 1: API serving the built UI, on a throwaway data dir
cd frontend && npm ci && npm run build && cd ../backend
export DL_VAR_DIR=/tmp/dl-e2e DL_ADMIN_PASSWORD=DefectLens-e2e-2026      # PowerShell: $env:DL_VAR_DIR="..."
defectlens init-db && defectlens bootstrap-demo
uvicorn app.main:app --port 8000

# terminal 2
cd frontend
npx playwright install chromium          # Linux may need: sudo npx playwright install-deps chromium
E2E_ADMIN_PASSWORD=DefectLens-e2e-2026 npx playwright test
```

On Windows (tested):

```powershell
$env:DL_VAR_DIR="$env:TEMP\dl-e2e"; $env:DL_ADMIN_PASSWORD="DefectLens-e2e-2026"
cd backend; .\.venv\Scripts\Activate.ps1
defectlens init-db; defectlens bootstrap-demo
uvicorn app.main:app --port 8000
# terminal 2
cd frontend
$env:E2E_BASE_URL="http://127.0.0.1:8000"; $env:E2E_ADMIN_PASSWORD="DefectLens-e2e-2026"
npx playwright test
```

The suite creates its own inspector, viewer and logout users through the API, and re-uses sessions to stay
under the login rate limit. If you run E2E and then the screenshot capture straight away, wait about a
minute or you'll get `429 Too many login attempts`: that's the rate limiter working.

To reproduce the screenshots on a real dataset, seed genuine activity first (the script calls the same
public API, so every inspection is a real model output; it does **not** fabricate reviews):

```bash
python scripts/seed_screenshots.py --base-url http://127.0.0.1:8000 \
    --product visa-pcb1 --dataset-dir var/datasets/visa-pcb1 --evaluate --robustness
# PowerShell: $env:SCREENSHOTS="1"; npx playwright test e2e/screenshots.spec.ts
SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts    # from frontend/
```

## Latest results

Final verification run on **Windows 11 Home (build 26300), Python 3.14.8, Node 24.21.0, PyTorch 2.14.1+cpu,
Intel i7-12700H (14 cores / 20 threads), 15.6 GB RAM, no GPU**, on 2026-10-19. The earlier Linux
(Python 3.13, Node 20) results are kept below where they add information.

| Check | Result |
|---|---|
| ruff, mypy (Python 3.14, 65 source files) | clean |
| Backend fast suite, Python 3.14 (lock) | **105 passed** |
| Backend slow suite, Python 3.14 | **2 passed** in 49 s (real PatchCore/PaDiM training + ONNX parity) |
| Frontend `npm run typecheck` + `npm run build` (Node 24.21) | OK |
| Playwright E2E, Chromium, against the built UI and real backend | **18 passed, 1 skipped** (screenshot capture, by design) |
| Documentation screenshot capture | **17 UI screenshots** regenerated from the running app; the 18th image is the benchmark plot produced by the reproducibility run |
| Symbol, API-route and API-doc consistency scripts | OK |
| Ollama provider (local, `gemma3:1b`, `http://localhost:11434/v1`) | model list, connection test (`reply: OK`, 2.1 s) and a real inspection summary (532 input / 260 output tokens) succeeded |
| Linux/Python 3.13 results from the previous environment | 105 fast + 2 slow passed, ruff/mypy clean, 18 E2E passed |
| Python 3.11 (pyproject ranges, Linux) | fast suite, ruff and mypy passed |
| `pip-audit` (lock + torch/torchvision) | no known vulnerabilities on Linux (after upgrading setuptools 78.1.0 → 84.0.0) and on Windows/Python 3.14 |
| `npm audit --omit=dev` | 0 vulnerabilities |
| gitleaks (files to be committed, Linux) | no leaks (two deliberate fake test keys allowlisted by exact value) |
| Lock install on Windows / Python 3.14 | yes: full install from `requirements.lock`, then executed (not just resolved) |

## Benchmark reproduction runs

Model benchmarks are not unit tests, but they were re-run on the rebuilt environment as a check:

| Run | Result |
|---|---|
| `prepare-visa --download … pcb1` then `train-visa --category pcb1` (3 models), Windows/Python 3.14 | fresh download and fresh database; training 268 s (PatchCore 148 s, PaDiM 105 s, PixelStats 15 s). Every pcb1 metric in `MODELS.md` (AUROC + CI, AP, F1, recall, FRR, auto-PASS miss, review rate, pixel AUROC, AUPRO) matched to 3 decimals |
| Same pcb1 run on Linux/Python 3.13 (2 vCPU) | `prepare-visa` 56 s + `train-visa` 214 s; identical non-latency metrics |
| In-app Model lab benchmark (3 models, 130 test images), Windows | 66 s; AUROC .839/.914/.883 for PixelStats/PaDiM/PatchCore, identical to the CLI |
| In-app Robustness lab (19 degradations × 3 severities, 40 test images), Windows | 272 s; robustness score **0.8413**, clean AUROC 0.925, recall 0.55, FRR 0.00 — identical to the Linux run (368 s) |
| `scripts/robustness_policy_sweep.py visa-pcb1 visa-pcb1-patchcore`, Linux | 351 s; 0 of 1,140 defective-image decisions auto-PASSed under degradation (1 of 20 on clean images) |
| Sample seeding for screenshots (`scripts/seed_screenshots.py`) | 130 real VisA pcb1 test images inspected through the public API: 48 PASS / 12 REVIEW / 70 FAIL; then the benchmark and robustness jobs above |

## Not tested (no hardware or service available)

- **Docker** (no daemon in the sandbox): the Dockerfile/Compose files are untested. CI builds the image and
  checks container health on the first push.
- **GPU / CUDA**: code paths exist (`DL_DEVICE`), but no GPU was available.
- **Physical USB/IP cameras**: the adapters are covered with simulated frames (corrupt and blank frames
  fail safe). The browser webcam page was exercised only without a real camera.
- **PostgreSQL**: driver and migrations are present; all runs used SQLite.
- **External cloud LLM providers** (OpenAI, Anthropic, OpenRouter, other paid OpenAI-compatible endpoints):
  connection handling, masking and encryption are tested with fakes; no paid API key was used. The local
  Ollama path was tested end to end.
