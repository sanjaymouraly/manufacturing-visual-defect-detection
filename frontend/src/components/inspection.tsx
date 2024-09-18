import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Eye, EyeOff, ImageOff, ZoomIn, ZoomOut } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/api";
import { num, prob } from "../lib/format";
import type { Inspection, Job, Region } from "../lib/types";
import { Button, ProgressBar, StatusBadge } from "./ui";

/** Raw anomaly score on a scale with the PASS / REVIEW / FAIL bands of the model's thresholds. */
export function ScoreGauge({ score, pass, fail }: { score: number | null; pass: number | null; fail: number | null }) {
  if (score === null || pass === null || fail === null) return <p className="muted text-sm">No score (inspection could not be performed).</p>;
  const max = Math.max(fail * 1.6, score * 1.05);
  const min = Math.min(pass * 0.6, score * 0.95, 0);
  const pos = (v: number) => `${((v - min) / (max - min)) * 100}%`;
  return (
    <div aria-label={`Score ${score.toFixed(3)}; pass below ${pass.toFixed(3)}; fail at or above ${fail.toFixed(3)}`}>
      <div className="relative h-3 overflow-hidden rounded-full">
        <div className="absolute inset-y-0 left-0 bg-emerald-400/70" style={{ width: pos(pass) }} />
        <div className="absolute inset-y-0 bg-amber-400/70" style={{ left: pos(pass), width: `calc(${pos(fail)} - ${pos(pass)})` }} />
        <div className="absolute inset-y-0 right-0 bg-red-400/70" style={{ left: pos(fail) }} />
      </div>
      <div className="relative h-5">
        <div className="absolute -top-4 h-5 w-0.5 bg-slate-900 dark:bg-white" style={{ left: pos(score) }} />
      </div>
      <div className="muted flex justify-between text-[11px]">
        <span>PASS &lt; {num(pass)}</span>
        <span className="font-semibold text-slate-900 dark:text-white">score {num(score)}</span>
        <span>FAIL ≥ {num(fail)}</span>
      </div>
    </div>
  );
}

type Layer = "display" | "overlay" | "heatmap" | "mask";

/** Original image with an adjustable heatmap overlay, predicted regions, and optional reviewer boxes. */
export function ImageViewer({ insp, extraBoxes, className }: { insp: Inspection; extraBoxes?: number[][]; className?: string }) {
  const [layer, setLayer] = useState<Layer>("heatmap");
  const [opacity, setOpacity] = useState(0.55);
  const [showBoxes, setShowBoxes] = useState(true);
  const [zoom, setZoom] = useState(1);
  const imgs = insp.images;
  if (insp.images_purged || !imgs.display)
    return (
      <div className={clsx("flex aspect-[4/3] flex-col items-center justify-center gap-2 rounded-lg bg-slate-100 dark:bg-slate-800", className)}>
        <ImageOff className="size-8 text-slate-400" />
        <p className="muted text-sm">{insp.images_purged ? "Images removed by the retention policy (record kept)." : "No image stored."}</p>
      </div>
    );
  const w = insp.width ?? 1;
  const h = insp.height ?? 1;
  const regions: Region[] = insp.regions ?? [];
  const layers: { id: Layer; label: string; available: boolean }[] = [
    { id: "heatmap", label: "Heatmap", available: !!imgs.heatmap },
    { id: "overlay", label: "Overlay", available: !!imgs.overlay },
    { id: "mask", label: "Mask", available: !!imgs.mask },
    { id: "display", label: "Original", available: true },
  ];
  return (
    <div className={className}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg border border-slate-300 p-0.5 dark:border-slate-700" role="radiogroup" aria-label="Visualization layer">
          {layers.filter((l) => l.available).map((l) => (
            <button key={l.id} role="radio" aria-checked={layer === l.id} onClick={() => setLayer(l.id)}
              className={clsx("rounded-md px-2.5 py-1 text-xs font-medium", layer === l.id ? "bg-brand-700 text-white" : "hover:bg-slate-100 dark:hover:bg-slate-800")}>{l.label}</button>
          ))}
        </div>
        {layer !== "display" && (
          <label className="flex items-center gap-2 text-xs">
            <span className="muted">Opacity</span>
            <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Overlay opacity" className="w-28" />
            <span className="w-8 tabular-nums">{Math.round(opacity * 100)}%</span>
          </label>
        )}
        <Button size="sm" variant="ghost" onClick={() => setShowBoxes((s) => !s)} aria-pressed={showBoxes}>
          {showBoxes ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />} Regions
        </Button>
        <div className="ml-auto flex gap-1">
          <Button size="sm" variant="ghost" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(1, z - 0.5))}><ZoomOut className="size-3.5" /></Button>
          <Button size="sm" variant="ghost" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(4, z + 0.5))}><ZoomIn className="size-3.5" /></Button>
        </div>
      </div>
      <div className="overflow-auto rounded-lg bg-slate-900" style={{ maxHeight: "70vh" }}>
        <div className="relative mx-auto" style={{ width: `${zoom * 100}%`, aspectRatio: `${w} / ${h}` }}>
          <img src={imgs.display} alt={`Inspected image ${insp.filename}`} className="absolute inset-0 size-full object-contain" />
          {layer !== "display" && imgs[layer] && (
            <img src={imgs[layer]!} alt={`${layer} visualization`} className="absolute inset-0 size-full object-contain" style={{ opacity, mixBlendMode: layer === "mask" ? "screen" : "normal" }} />
          )}
          {showBoxes && (
            <svg className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" aria-hidden>
              {regions.map((r, i) => (
                <g key={i}>
                  <rect x={r.bbox[0]} y={r.bbox[1]} width={r.bbox[2]} height={r.bbox[3]} fill="none" stroke="#f43f5e" strokeWidth={Math.max(2, w / 300)} />
                  <text x={r.bbox[0]} y={Math.max(r.bbox[1] - 4, 12)} fill="#f43f5e" fontSize={Math.max(12, w / 45)} fontWeight="bold">#{i + 1}</text>
                </g>
              ))}
              {(extraBoxes ?? []).map((b, i) => (
                <rect key={`r${i}`} x={b[0]} y={b[1]} width={b[2]} height={b[3]} fill="rgba(56,189,248,0.15)" stroke="#38bdf8" strokeDasharray="6 4" strokeWidth={Math.max(2, w / 300)} />
              ))}
            </svg>
          )}
        </div>
      </div>
      <p className="muted mt-1 text-[11px]">
        Heatmap colours use a fixed per-model scale (blue = typical of normal training images, red = highly unusual). Red boxes: regions above the
        segmentation threshold, or the peak area (marked ≈ in the region list) when none reached it{extraBoxes?.length ? "; dashed blue boxes: inspector-marked regions" : ""}.
      </p>
    </div>
  );
}

export function ProbabilityLine({ insp }: { insp: Inspection }) {
  return (
    <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
      <div><p className="muted text-xs">Raw anomaly score</p><p className="font-semibold tabular-nums">{num(insp.score)}</p><p className="muted text-[11px]">unitless, not a probability</p></div>
      <div><p className="muted text-xs">Normalised (1.0 = reject threshold)</p><p className="font-semibold tabular-nums">{num(insp.normalized_score, 2)}</p></div>
      <div><p className="muted text-xs">Calibrated defect probability</p><p className="font-semibold tabular-nums">{insp.calibrated_probability === null ? "not calibrated" : prob(insp.calibrated_probability)}</p><p className="muted text-[11px]">Platt fit on validation split</p></div>
      <div><p className="muted text-xs">OOD distance</p><p className="font-semibold tabular-nums">{num(insp.ood_score, 1)}</p><p className="muted text-[11px]">image-level novelty</p></div>
    </div>
  );
}

/** Polls a background job until it finishes. */
export function useJob(jobId: string | null | undefined) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api<Job>(`/api/jobs/${jobId}`),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data && ["succeeded", "failed", "cancelled"].includes(q.state.data.status) ? false : 1000),
  });
}

export function JobStatus({ jobId, onCancel }: { jobId: string; onCancel?: () => void }) {
  const { data: j } = useJob(jobId);
  if (!j) return null;
  const active = j.status === "queued" || j.status === "running";
  return (
    <div className="space-y-2 rounded-lg border border-slate-200 p-3 dark:border-slate-800" aria-live="polite">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <StatusBadge status={j.status} />
        <span className="font-medium">{j.type.replace(/_/g, " ")}</span>
        <span className="muted truncate">{j.message}</span>
        {active && onCancel && <Button size="sm" variant="ghost" className="ml-auto" onClick={onCancel} disabled={j.cancel_requested}>{j.cancel_requested ? "Cancelling…" : "Cancel"}</Button>}
      </div>
      {active && <ProgressBar value={j.progress} label={`${j.type} progress`} />}
      {j.error && <p className="text-sm text-red-600 dark:text-red-400">{j.error}</p>}
    </div>
  );
}
