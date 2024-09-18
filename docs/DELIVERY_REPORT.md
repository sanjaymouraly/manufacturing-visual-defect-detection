# Delivery report: DefectLens

_Date: 2026-10-19. Final environment: Windows 11 Home (build 26300) x64, Intel i7-12700H (14 cores / 20
threads), 15.6 GB RAM, no GPU, Python 3.14.8, Node 24.21.0, PyTorch 2.14.1 CPU. The project was previously
built and tested on Debian 13 (2 vCPU, ~2 GB RAM, Python 3.13.16, Node 20.20.2); those results are kept
where they add information. No Docker daemon was available in either environment._

## 1. What was delivered

A complete, self-hosted manufacturing visual-inspection platform (not a prototype):

- **Backend** (`backend/app`): FastAPI + Pydantic, SQLAlchemy 2 + Alembic (SQLite dev / PostgreSQL),
  85 API operations, DB-backed background jobs (progress, cancel, retry, crash recovery), RBAC, audit
  chain, rate limiting, upload/ZIP safety, retention, backup/restore, health and metrics, and an admin CLI
  (`defectlens`).
- **ML library** (`backend/defectml`): PatchCore, PaDiM, PixelStats, defect-type classifier; dataset
  importers (synthetic, VisA, folder), validation and quality reports, leakage-free splits; evaluation
  (image/pixel/region metrics with CIs, calibration, latency/memory), 19-type robustness bench,
  domain-shift experiment, ONNX export with parity check; CLI (`defectml`).
- **Web UI** (`frontend`): React 19 + TypeScript + Vite + Tailwind. Pages: dashboard, inspect, visual
  explanation (layers, opacity, zoom, regions), batch (multi-file/ZIP, progress, CSV export), camera
  (webcam interval capture, server cameras), review queue and detail (region drawing), history (filters),
  model lab (compare, approve, activate, rollback, train, ONNX), robustness lab, analytics/reports,
  datasets and feedback, jobs, and settings (policy/thresholds, products, cameras, users, AI providers,
  privacy, retention, audit). Dark/light themes, keyboard and screen-reader labels, responsive layout.
- **Docs**: README (Linux, Windows, Docker, GPU), architecture, API (generated route table), dataset card,
  model card and evaluation report, testing guide, SECURITY, CONTRIBUTING, LICENSE (Apache-2.0) + NOTICE
  (data and weights terms kept separate), `.env.example`, 18 screenshots (17 UI captures plus a benchmark
  plot).
- **Ops**: GitHub Actions CI (lint/type/test on py3.13 + py3.11, slow suite, frontend build, Playwright
  E2E against a freshly bootstrapped backend, pip-audit / npm audit / gitleaks, Docker build + container
  health), Dockerfile + Compose (CPU, PostgreSQL, GPU overrides), `.gitignore` / `.dockerignore`,
  `.gitleaks.toml`.

Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md). The decision policy is conservative by design: FAIL
at or above the rejection threshold; PASS only below the acceptance threshold with no quality, OOD or
model-disagreement flags; everything else, including every error, goes to REVIEW. Models change only
through explicit, audited admin approval, never by automatic retraining.

## 2. Data and licensing

- **Bundled**: synthetic demo data generator and 12 synthetic samples (Apache-2.0, clearly labelled
  SYNTHETIC).
- **VisA** (CC BY 4.0) for real benchmarks. Downloaded by the user, never redistributed, attribution in
  `NOTICE`, the dataset card and the screenshot index. Defect types come from VisA's textual annotations,
  never from masks.
- **MVTec AD not used** (non-commercial license). ImageNet ResNet-18 weights are downloaded at runtime and
  not redistributed (ImageNet terms flagged in `NOTICE`).
- Details: [`DATASETS.md`](DATASETS.md).

## 3. Model results (held-out test split, thresholds from validation only)

| VisA category (n) | Best model here | Image AUROC [95 % CI] | Auto-PASS miss | Review rate | Pixel AUROC |
|---|---|---|---|---|---|
| pcb1 (130) | PaDiM | 0.914 [0.860–0.958] | 3.1 % | 9.2 % | 0.988 |
| cashew (97) | PatchCore | 0.975 [0.945–0.996] | 3.1 % | 24.7 % | 0.992 |
| capsules (104) | PatchCore | 0.739 [0.649–0.828] | 3.1 % | 73.1 % | 0.964 |

**Reproducibility:** pcb1 was re-downloaded, re-prepared and retrained from scratch on Linux, then again
on Windows 11 / Python 3.14 with a fresh dataset and database. Every pcb1 metric in the table (AUROC and
CI, AP, F1, recall, FRR, auto-PASS miss, review rate, pixel AUROC, AUPRO) matched to three decimals on
both runs; only latency changed with the hardware (38.9 / 93.2 / 11.5 ms per image for PaDiM / PatchCore /
PixelStats on the Windows machine, versus 32 / 144 / 2.4 ms on the 2-vCPU VM). The synthetic demo rows
were also regenerated on Windows and matched the recorded values.

**Robustness** (pcb1 PatchCore, all 19 corruptions × 3 severities, 40 test images): overall score **0.8413**
on both machines (Linux 368 s; Windows in-app job 272 s), with clean AUROC 0.925, recall 0.55 and FRR 0.00.
It is weak under motion/defocus blur, occlusion, resolution loss and 180° rotation. Most corruptions fail
toward false rejects. The exception is 180° rotation, where recall at the fail threshold drops to 0.40.
Even so, under the full decision policy **no defective image was auto-PASSed in any of the 57 conditions**
(0 of 1,140 decisions: 964 FAIL, 176 REVIEW).

Full table, latency, memory, ONNX parity, robustness, domain shift (a lighting change makes
every good part fail, which means retraining is needed) and caveats: [`MODELS.md`](MODELS.md). These
ResNet-18/224 px results are below published WideResNet-50 numbers. That trade-off was chosen for 2 GB
CPU machines.

## 4. Testing performed

| Check | Result |
|---|---|
| Backend fast suite (unit + API integration) | **105 passed** on Python 3.14/Windows and on Python 3.13/Linux; also passed on Python 3.11 (Linux) |
| Backend slow suite (real training + ONNX parity) | **2 passed** (49 s on Windows) |
| Playwright E2E (Chromium, real backend) | **18 passed**, 1 skipped by design (screenshot capture), final run on Windows against the rebuilt UI |
| ruff, mypy | clean (Python 3.14/Windows; previously 3.13 and 3.11/Linux) |
| Frontend typecheck + production build | OK (Node 24.21/Windows; previously Node 20/Linux) |
| Symbol, API-route and API-doc consistency scripts | OK |
| pip-audit, npm audit, gitleaks | pip-audit and `npm audit --omit=dev` clean on Linux and Windows; gitleaks clean on the Linux run (run again locally if the tool is installed; not installed in the Windows sandbox) |
| Clean rebuilds | Complete environment rebuilt from the README commands on Debian and Windows; demo and VisA pcb1 metrics reproduced to 3 decimals |
| Robustness benchmark | 19 degradations × 3 severities run on both machines; 0.8413 reproduced; full-policy decision sweep over the same 57 conditions (`scripts/robustness_policy_sweep.py`, Linux) |
| Local LLM (Ollama) | provider API listed `gemma3:1b` and `qwen3.5:4b`, connection test returned `OK`, and a real inspection summary was generated (532 input / 260 output tokens) |
| Screenshots | 17 UI screenshots regenerated from the running app on Windows, plus the benchmark plot |

Not testable here (stated everywhere it matters): Docker (no daemon), GPU, physical cameras, PostgreSQL,
external cloud LLM APIs.

## 5. Problems found by testing and fixed

The test-and-review loop found real defects. All are fixed and covered by tests where possible:

- **Logout/login** left a stale user in the UI (query cache cleared incorrectly); found by E2E.
- **Batch upload** read the file list after resetting the input, so uploads were empty; found by E2E.
- **Review form** sent severity `none`, which the API rejected with 422; found by E2E.
- **Viewers saw the Settings navigation** (the server already blocked them); found by E2E.
- **History search** fired on every keystroke and is now submit-based; the provider delete button lacked
  an accessible name; found by E2E.
- **ONNX export mismatch**: torch reflect padding vs SciPy symmetric padding, plus PatchCore argmin
  handling. Found by the slow parity test; parity is now ≤ 2.4e-6.
- **ZIP-bomb check order**: the compression-ratio check now runs before the size check; found by unit test.
- **CSP vs inline theme script**: moved to `theme-init.js`, so the strict CSP holds.
- **F1-optimal thresholds were degenerate** on small validation sets (FRR up to 0.95), so thresholds now
  use Youden-J.
- **`DL_CORS_ORIGINS=a,b` crashed startup** (the settings library JSON-decoded it). It now accepts a
  comma list or a JSON list.
- **No change-password UI** and no default-password warning existed, although the API supported password
  changes. Added a dialog, a banner and a `password_change_required` flag (with migration and tests).
- **Migration 0002 failed on populated SQLite databases** (table rebuild violated foreign keys). It now
  uses native ADD/DROP COLUMN, and a new test migrates a populated DB with FKs enforced. The test fails
  on the old version, confirming it catches the bug.
- **FAIL results without any box**: the pixel threshold could exceed the image score. FAIL/REVIEW now get
  an approximate peak box, flagged as such in the UI and API.
- **Readiness probe** reported storage "ok" when the directory merely existed. It now performs a real
  write.
- **`defectml` CLI wrote datasets to the wrong directory** when run from `backend/`, so the app couldn't
  see them. It now uses the app's configured directories and accepts bare dataset/model names.
- **`pip install -e . --no-deps` failed on a clean Windows checkout**: `backend/pyproject.toml` pointed
  `readme` at `../README.md`, outside the backend source tree. It now uses a backend-local README.
- **mypy failed on Windows** because `resource.RUSAGE_SELF` does not exist there; the POSIX-only call is
  now explicitly guarded (peak memory is reported as `null` on Windows).
- **Table cards were double-padded**: `Card` combined its default `p-4` with `bodyClass="p-0"`, and the
  later utility won in the generated CSS. This made wide tables overflow their card (visible in the
  robustness decision-flip table). `bodyClass` now replaces the default padding.
- **The evaluation comparison modal clipped its right-hand columns**; wide modals now use the available
  viewport width instead of a fixed 896 px, so all 16 metric columns are visible at 1440 px.
- **Dependency issues**: setuptools 78.1.0 had two advisories (now 84.0.0); `uvloop` would break Windows
  installs (now platform-marked); the lock file was originally documented for Python 3.13 only (it now
  installs and is tested on 3.13/3.14, with pyproject ranges for 3.11/3.12); a mypy error appeared only
  on 3.11; the CUDA wheel index `cu124` had no torch 2.14.1 build (switched to `cu126`).
- **Documentation claims** were checked against code and real output; several wrong defaults and example
  values were corrected before publication.

## 6. Known limitations

- Accuracy on hard categories (capsules) is modest; many items go to REVIEW by design.
- Sensitive to motion/defocus blur, occlusion, 180° pose changes and lighting changes. Retrain after any optics or lighting change.
- Rate limits and the default job worker are per process, so run one API process (or an external worker
  plus proxy-level limits).
- No SSO/LDAP/MFA. Single-site deployment model.
- **Optional LLM output can still be wrong.** In the local Ollama smoke test with `gemma3:1b`, the summary
  correctly repeated the decision, score, threshold and regions but also misread a model name as a defect
  type and described localisation boxes as "image quality issues". The UI labels every AI text as
  model-generated, shows a deterministic template fallback, and never lets LLM text change a decision.
  Use a larger model and review the output before sharing it.
- Windows does not expose the POSIX peak-RSS counter, so `peak_memory_mb` is `null` in metrics recorded
  on Windows; the Linux figures remain in the model card.
- The capsules and cashew rows were not re-run after the last sandbox reset; their model artifacts were
  lost after the metrics were recorded. pcb1 was regenerated and matched exactly on both Linux and
  Windows. All of them are reproducible with the documented commands (about 1.8 GB download plus a few
  minutes of CPU training per category).
- Robustness was measured on one category (pcb1) with one model (PatchCore), on 40 images. The
  "0 auto-PASS" result covers the same 20 defective boards under 57 transformations. It is not an
  escape-rate estimate.

## 7. Start-up commands

**Windows 11 (the final tested environment; PowerShell):**

```powershell
git clone https://github.com/<you>/defectlens.git; cd defectlens
Copy-Item .env.example .env                 # edit .env: set DL_ADMIN_PASSWORD (12+ characters)

cd backend
py -3.14 -m venv .venv                      # py -3.13 also works
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
pip install -r requirements.lock; pip install -e . --no-deps

cd ..\frontend; npm ci; npm run build; cd ..\backend
defectlens init-db
defectlens bootstrap-demo                   # optional synthetic demo, 6 small models
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> and sign in as `admin` with the password from `.env`. If PowerShell blocks
the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**Linux (previous tested environment; results retained where they add information):**

```bash
git clone https://github.com/<you>/defectlens.git && cd defectlens
cp .env.example .env                        # set DL_ADMIN_PASSWORD

cd backend && python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
pip install -r requirements.lock && pip install -e . --no-deps
cd ../frontend && npm ci && npm run build && cd ../backend
defectlens init-db && defectlens bootstrap-demo
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`DL_ADMIN_PASSWORD` is only used on first start. Health check: `curl http://localhost:8000/api/health/ready`.
Docker commands are in the README; the image was not built locally (no daemon).

## 8. Screenshots

`docs/images/` holds 18 PNGs: 17 UI screenshots captured from the running app by Playwright, plus one
benchmark plot. The UI screenshots are `dashboard.png`, `dashboard-dark.png`, `inspect-result.png`,
`heatmap.png`, `overlay.png`, `batch.png`, `review-queue.png`, `review-detail.png`, `history.png`,
`model-lab.png`, `model-comparison.png`, `model-detail.png`, `robustness.png`, `analytics.png`,
`datasets.png`, `settings-policy.png`, `settings-providers.png`. The plot is
`robustness-pcb1-19-degradations.png` (produced by `defectml robustness`, not a UI capture). Index and VisA
attribution: `docs/images/README.md`.

## 9. Publishing to GitHub

The delivered workspace already contains a `main` branch with the initial commit `DefectLens: initial
public release` and a clean working tree. To publish it:

```bash
cd defectlens
git status --short          # should print nothing
# optional local secret scan if gitleaks is installed:
gitleaks git . --config .gitleaks.toml

# create an empty repository on GitHub (no README/license), then:
git remote add origin https://github.com/<you>/defectlens.git
git push -u origin main
```

If you copied the sources without the `.git` directory, initialise it first:

```bash
git init -b main
git add .
git status --short | grep -E "(^|/)(var|node_modules|\.venv|dist)/|\.env$|\.db$|\.npz$|\.onnx$" \
  && echo "STOP: something that must not be committed is staged"       # should print nothing
git commit -m "DefectLens: initial public release"
```

After pushing:

1. Replace `<you>` in `README.md` with your GitHub user or organisation.
2. Open *Actions* and check the first CI run. The `docker` job is the first real build of the image; fix
   anything it reports.
3. *Settings → Code security*: enable Dependabot alerts, secret scanning and **private vulnerability
   reporting** (`SECURITY.md` points to it).
4. *Settings → Branches*: protect `main` and require the `backend`, `frontend`, `e2e` and `security` jobs.
5. Optional: add a CI status badge to the README once the repository URL is final.

The repository holds 187 tracked files (6.65 MB, mostly the screenshots). It contains no datasets,
weights, databases or secrets.
