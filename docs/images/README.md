# Screenshots

Captured from the running application by Playwright (`frontend/e2e/screenshots.spec.ts`, run with
`npm run screenshots`). They show real UI states produced by real inspections, not mock-ups.

| File | Shows |
|---|---|
| `dashboard.png`, `dashboard-dark.png` | Dashboard (light / dark theme): decision mix, recent inspections, review backlog |
| `inspect-result.png` | Single inspection result: decision, reasons, score bar, calibrated probability, OOD distance, regions, quality checks, traceability |
| `heatmap.png`, `overlay.png` | Visual explanation: anomaly heatmap and overlay with opacity slider and region boxes |
| `batch.png` | Batch inspection with per-image results and CSV export |
| `review-queue.png`, `review-detail.png` | Human review queue and review form (label, defect type, severity, region drawing, feedback) |
| `history.png` | Inspection history with filters and search |
| `model-lab.png`, `model-comparison.png`, `model-detail.png` | Model versions, side-by-side held-out comparison, model card with metrics |
| `robustness.png` | Robustness lab: the 19-degradation benchmark run on pcb1 PatchCore (run header, scores, metric-vs-severity chart, per-degradation table, decision flips) |
| `analytics.png` | Analytics and reports |
| `datasets.png` | Datasets, quality reports and feedback approval |
| `settings-policy.png`, `settings-providers.png` | Decision policy / thresholds; optional AI providers |
| `robustness-pcb1-19-degradations.png` | Benchmark plot (not a UI screenshot): AUROC heatmap and false-rejection curves of pcb1 PatchCore under 19 degradations × 3 severities, from `defectml robustness`; discussed in `docs/MODELS.md` |

## Image content and attribution

Several screenshots (inspection, heatmap/overlay, review, model and robustness views) show images from the
**VisA** dataset, category *pcb1*: Zou et al., "SPot-the-Difference Self-Supervised Pre-training for
Anomaly Detection and Segmentation", ECCV 2022 (https://github.com/amazon-science/spot-diff), licensed
**CC BY 4.0**. The images were downscaled and overlaid with model output (heatmaps, boxes) for these
screenshots. Other screenshots show the project's own synthetic demo images (Apache-2.0).
