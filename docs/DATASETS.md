# Dataset card

DefectLens never ships third-party images. It ships a **synthetic demo generator** plus importers that
fetch or read datasets on your machine, record their license and provenance, and validate them.

## 1. Synthetic demo data (bundled, Apache-2.0)

| | |
|---|---|
| Products | `demo-pcb` (circuit-board-like boards: traces, pads, components), `demo-metal` (machined ring / bore) |
| Generator | `defectml/data/synthetic.py`, deterministic (seed 42). `defectml gen-synthetic` or `defectlens bootstrap-demo` |
| Size per product | 160 normal + 60 anomalous images, split by simulated production batch (group-aware): 15 % of normals to validation, 20 % to test, anomalies shared between validation and test (n = 68 test images per product) |
| Defects | PCB: missing component, solder bridge, contamination, scratch. Metal: scratch, dent, contamination, crack. Each batch has its own lighting offset |
| Masks / types | Pixel masks and defect types are produced *by the generator*, so they are genuine ground truth for this data |
| Bundled files | `samples/` holds 12 example images (7 PCB, 5 metal) for quick manual tests |
| Labelling | Every synthetic dataset, model and metric carries `synthetic_data: true`, and the UI marks it **SYNTHETIC** |

**Limitations.** The demo images are far easier than real production images: uniform lighting, perfect
alignment, high-contrast defects. Perfect demo scores (AUROC 1.000) say **nothing** about real-world
performance. The demo exists to exercise the whole workflow offline.

## 2. VisA (real data; downloaded by the user, CC BY 4.0)

| | |
|---|---|
| Source | Amazon Science, *SPot-the-difference* (Zou et al., ECCV 2022), https://github.com/amazon-science/spot-diff |
| License | **CC BY 4.0**, attribution required (see `NOTICE`). The code repo is Apache-2.0 |
| Download | `defectml prepare-visa --download --accept-license --categories pcb1 capsules cashew` (the full archive is about 1.8 GB; only the selected categories are extracted). `--tar FILE` or `--raw DIR` reuse an existing copy |
| Preprocessing | Images resized so the longest side ≤ 512 px (`--max-side`), masks resized with nearest-neighbour, file hashes recorded in the manifest |
| Labels | Image label from the official `split_csv/1cls.csv`. **Defect types come from the textual labels in each category's `image_anno.csv`** (genuine annotations), never inferred from masks |
| Categories used here | `pcb1` (1004 normal / 100 anomalous), `capsules` (602 / 100), `cashew` (500 / 100) |

### Splits (leakage-free, reproducible)

VisA's official split has train (normal only) and test (normal + anomalous). DefectLens keeps the official
training set and divides the official test set deterministically (SHA-256 of seed + file name) into
**validation** (threshold selection, calibration) and **test** (reporting only). Thresholds are never tuned
on test.

| Category | train normal | val normal | val anomalous | test normal | test anomalous |
|---|---|---|---|---|---|
| pcb1 | 904 | 35 | 35 | 65 | 65 |
| capsules | 542 | 21 | 35 | 39 | 65 |
| cashew | 450 | 18 | 35 | 32 | 65 |

pcb1 has four annotated defect classes. Of its 100 anomalous images, 91 carry one label and 9 carry
several (e.g. `melt|scratch`), so the per-class totals sum to 110: melt 54, scratch 21, missing 20, bent 15.
The test split's 65 anomalies contain melt 34, bent 10, scratch 9, missing 6 and 6 multi-label images.
In per-type recall, a multi-label image counts toward each of its classes (never reduced to one class).

VisA has no batch, session or camera identifiers, so group-aware splitting (used for custom data that has
them) is not possible. Near-duplicates across splits are detected and reported instead.

## 3. Custom data (your own images)

`defectml import-folder SRC --name NAME --product KEY --license "..."` (or the API
`POST /api/datasets/import`). Two layouts are recognised:

```
Simple                                         MVTec-style
SRC/                                           SRC/
  normal/      (or good/, ok/)                   train/good/*.png
  anomaly/     (or defective/, bad/, ng/)        test/good/*.png
  anomaly/<type>/  optional: genuine types       test/<defect_type>/*.png
  masks/<stem>.png optional, non-zero = defect   ground_truth/<defect_type>/<stem>_mask.png
  groups.csv   optional: filename,group  (batch/lot/camera; a group never spans two splits)
```

Images are safely decoded and re-encoded as PNG. Defect types are taken only from explicit folder names.
State the license or source with `--license`; the manifest records it with SHA-256 hashes of every file.

## 4. Validation and quality report (`defectml validate`, also run on import)

For every dataset the report (`<dataset>/quality_report.json` plus a preview mosaic) contains:

- unreadable or corrupt files; format and dimension histograms
- mask checks: missing, unreadable, shape mismatch with the image, empty mask on an anomalous sample,
  non-empty mask on a normal sample
- exact duplicates (SHA-256) and near-duplicates (64-bit dHash, Hamming distance ≤ 3), with examples
- class balance (normal : anomalous ratio) and defect-type distribution
- split summary and a **leakage check** (the same hash or near-duplicate in two splits, or a group spanning splits)
- an overall `valid` flag: no read errors, no mask issues, leakage-free

## 5. Feedback datasets (human-verified, versioned)

Reviewed inspections can be proposed for feedback by inspectors. Admins approve or reject each item.
*Build feedback version* creates a new, versioned dataset `<base>-fb-vN`: the base dataset plus all
approved, not-yet-used feedback samples. The base is never modified. Each added image records its source
inspection, review and reviewer (`source_id`) and the corrected label/type. Every sample can be used in only
one version, and rejected items stay in the audit trail but are excluded. Feedback datasets are only used when an admin
explicitly trains a new candidate model with them.

## 6. Not used: MVTec AD

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial). DefectLens does not download or use it by
default. You may import it yourself as a folder dataset for non-commercial research; models trained on it
inherit the restriction.
