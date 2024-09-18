import { useMutation } from "@tanstack/react-query";
import { AlertTriangle, Bot, ClipboardCheck, Copy, FileDown, ShieldAlert } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api, download } from "../lib/api";
import { useAuth } from "../lib/auth";
import { dt, ms, num, pct, titleCase } from "../lib/format";
import type { AiText, Inspection } from "../lib/types";
import { useToast } from "../lib/toast";
import { ImageViewer, ProbabilityLine, ScoreGauge } from "./inspection";
import { Badge, Button, Card, DecisionBadge, ErrorBox, KV, Note, SeverityBadge } from "./ui";

export function DecisionHeader({ insp }: { insp: Inspection }) {
  const tone = insp.decision === "PASS" ? "border-emerald-500" : insp.decision === "FAIL" ? "border-red-500" : "border-amber-500";
  return (
    <div className={`card border-l-4 p-4 ${tone}`}>
      <div className="flex flex-wrap items-center gap-3">
        <DecisionBadge decision={insp.decision} large />
        <SeverityBadge severity={insp.severity} />
        {insp.status === "error" && <Badge tone="red"><ShieldAlert className="size-3" /> inspection failed — routed to review</Badge>}
        {insp.review_status === "pending" && <Badge tone="amber">awaiting inspector review</Badge>}
        {insp.review_status === "reviewed" && <Badge tone="green">reviewed: {insp.final_label}</Badge>}
        {insp.duplicate && <Badge tone="blue">duplicate image — existing result returned</Badge>}
        <span className="muted ml-auto text-xs">{insp.model_name ?? "no model"} · {ms(insp.latency_ms)}</span>
      </div>
      <ul className="mt-3 space-y-1 text-sm">
        {insp.reasons.map((r, i) => <li key={i} className="flex gap-2"><span className="text-slate-400">•</span>{r}</li>)}
        {insp.error && <li className="flex gap-2 text-red-600 dark:text-red-400"><AlertTriangle className="mt-0.5 size-4 shrink-0" />{insp.error}</li>}
      </ul>
    </div>
  );
}

export default function InspectionResult({ insp, compact }: { insp: Inspection; compact?: boolean }) {
  const { can } = useAuth();
  const toast = useToast();
  const [summary, setSummary] = useState<AiText | null>(null);
  const sum = useMutation({
    mutationFn: () => api<AiText>(`/api/inspections/${insp.id}/summary`, { method: "POST" }),
    onSuccess: setSummary,
  });
  const lastReview = insp.reviews?.[insp.reviews.length - 1];
  return (
    <div className="space-y-4">
      <DecisionHeader insp={insp} />
      <div className="grid gap-4 xl:grid-cols-5">
        <Card title="Visual explanation" className="xl:col-span-3">
          <ImageViewer insp={insp} extraBoxes={lastReview?.regions} />
        </Card>
        <div className="space-y-4 xl:col-span-2">
          <Card title="Score & confidence">
            <ScoreGauge score={insp.score} pass={insp.pass_threshold} fail={insp.fail_threshold} />
            <div className="mt-4"><ProbabilityLine insp={insp} /></div>
          </Card>
          {insp.defect_type && (
            <Card title="Defect type (classifier)">
              <p className="text-sm"><span className="font-semibold">{titleCase(insp.defect_type.defect_type)}</span> <span className="muted">({pct(insp.defect_type.probability)})</span></p>
              {insp.defect_type.top && <p className="muted mt-1 text-xs">{Object.entries(insp.defect_type.top).map(([k, v]) => `${k} ${pct(v, 0)}`).join(" · ")}</p>}
              <p className="muted mt-1 text-[11px]">{insp.defect_type.note}</p>
            </Card>
          )}
          {insp.secondary && (
            <Card title="Second-opinion model">
              {insp.secondary.error ? <p className="text-sm text-red-600">{insp.secondary.error}</p> :
                <p className="text-sm">{insp.secondary.model_name}: <DecisionBadge decision={insp.secondary.decision ?? "REVIEW"} /> score {num(insp.secondary.score ?? null)}</p>}
            </Card>
          )}
          {!compact && (
            <Card title="Traceability">
              <KV items={[
                ["Inspection ID", <span className="font-mono text-xs">{insp.id}</span>],
                ["Time", dt(insp.created_at)],
                ["Product", insp.product_key],
                ["Model", `${insp.model_name ?? "—"} (${insp.algorithm ?? "—"})`],
                ["Source", insp.source + (insp.camera_id ? ` · ${insp.camera_id}` : "")],
                ["Batch / station", `${insp.batch_code || "—"} / ${insp.station || "—"}`],
                ["External ref", insp.external_ref || "—"],
                ["File", insp.filename],
                ["Image SHA-256", <span className="font-mono text-[11px] break-all">{insp.image_sha256}</span>],
                ["Size", insp.width ? `${insp.width}×${insp.height}` : "—"],
              ]} />
            </Card>
          )}
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title={`Localized regions (${insp.regions?.length ?? insp.n_regions})`} bodyClass="p-0">
          {insp.regions && insp.regions.length > 0 ? (
            <table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">#</th><th className="th">Box (x, y, w, h)</th><th className="th">Area</th><th className="th">Peak score</th></tr></thead>
              <tbody>{insp.regions.map((r, i) => (
                <tr key={i} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td">{i + 1}</td><td className="td font-mono text-xs">{r.bbox.join(", ")}</td><td className="td">{pct(r.area_fraction, 2)}</td><td className="td tabular-nums">{num(r.max_score)}{r.approximate && <span className="ml-1 text-xs text-amber-700 dark:text-amber-300" title="Approximate (peak) localization">≈</span>}</td></tr>
              ))}</tbody></table>
          ) : <p className="muted p-4 text-sm">No region exceeded the segmentation threshold.</p>}
          {insp.regions?.some((r) => r.approximate) && (
            <p className="border-t border-slate-200 p-3 text-xs text-amber-700 dark:border-slate-800 dark:text-amber-300">
              Approximate localization: no area reached the segmentation threshold, so the box marks the peak of the anomaly map.
              Check the heatmap and the image itself before relying on it.
            </p>
          )}
        </Card>
        <Card title="Image quality checks">
          {insp.quality ? (
            <>
              <p className="mb-2 text-sm">{insp.quality.ok ? <Badge tone="green">acceptable</Badge> : <Badge tone="amber">issues found — result needs review</Badge>}</p>
              <KV items={[["Brightness", num(insp.quality.brightness, 1)], ["Contrast", num(insp.quality.contrast, 1)], ["Sharpness (Laplacian var.)", num(insp.quality.sharpness, 1)], ["Saturated pixels", pct(insp.quality.saturated_fraction)]]} />
              {insp.quality.issues.length > 0 && <ul className="mt-2 list-disc pl-5 text-sm text-amber-700 dark:text-amber-300">{insp.quality.issues.map((x) => <li key={x}>{x}</li>)}</ul>}
            </>
          ) : <p className="muted text-sm">Not available.</p>}
        </Card>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button onClick={() => download(`/api/inspections/${insp.id}/report.pdf`, `inspection-${insp.id.slice(0, 8)}.pdf`).catch((e) => toast("error", e.message))}><FileDown className="size-4" /> PDF report</Button>
        {can("admin", "inspector") && <Link to={`/reviews/${insp.id}`}><Button variant={insp.review_status === "pending" ? "warning" : "secondary"}><ClipboardCheck className="size-4" /> {insp.review_status === "pending" ? "Review now" : "Correct / review"}</Button></Link>}
        <Button onClick={() => { navigator.clipboard?.writeText(insp.id); toast("ok", "Inspection ID copied"); }}><Copy className="size-4" /> Copy ID</Button>
        <Button onClick={() => sum.mutate()} loading={sum.isPending}><Bot className="size-4" /> Explain result</Button>
      </div>
      <ErrorBox error={sum.error} title="Summary unavailable" />
      {summary && (
        <Card title={summary.model_generated ? "AI-generated summary" : "Summary (deterministic template)"} subtitle={summary.generated_by}>
          {summary.model_generated && <div className="mb-2"><Note tone="warn">Model-generated text from recorded inspection data only (no images were sent). It does not change the decision; verify against the data above.</Note></div>}
          <p className="text-sm whitespace-pre-wrap">{summary.text}</p>
          {summary.model_generated && <p className="muted mt-2 text-[11px]">tokens {summary.tokens_in ?? "?"} in / {summary.tokens_out ?? "?"} out · est. cost {summary.cost_usd === null || summary.cost_usd === undefined ? "unknown (no pricing configured)" : `$${summary.cost_usd.toFixed(5)}`}</p>}
        </Card>
      )}
    </div>
  );
}
