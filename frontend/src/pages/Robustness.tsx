import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, Play, Shuffle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { JobStatus, useJob } from "../components/inspection";
import ProductSelect from "../components/ProductSelect";
import { Badge, Button, Card, DecisionBadge, Empty, ErrorBox, Field, KV, Note, PageHeader, Spinner, Tabs } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { dt, num, pct, titleCase } from "../lib/format";
import { useModels } from "../lib/hooks";
import type { Job } from "../lib/types";

interface RobRow { degradation: string; severity: number; auroc: number | null; ap: number | null; recall: number | null; miss_rate: number | null; false_rejection_rate: number | null; f1: number | null; decision_flip_rate: number; mean_score_shift: number }
interface RobResult { model: string; algorithm: string; dataset: string; synthetic_data?: boolean; split: string; n: number; severities: number[]; clean: Omit<RobRow, "degradation" | "severity" | "decision_flip_rate" | "mean_score_shift">; rows: RobRow[]; robustness_score: number; robustness_by_degradation: Record<string, number>; worst_conditions: RobRow[]; decision_flips: { path: string; degradation: string; severity: number; label: string; clean_score: number; degraded_score: number }[]; seconds: number }
interface EvalRun { id: number; kind: string; product_key: string; dataset_name: string; created_at: string; model?: string | null; robustness_score?: number | null }
const PALETTE = ["#0d9488", "#ef4444", "#6366f1", "#f59e0b", "#10b981", "#ec4899", "#0ea5e9", "#84cc16", "#a855f7", "#f97316", "#14b8a6", "#64748b"];

function RobustnessView({ id }: { id: number }) {
  const q = useQuery({ queryKey: ["evaluation", id], queryFn: () => api<EvalRun & { result: RobResult; artifacts: string[] }>(`/api/evaluations/${id}`) });
  const [metric, setMetric] = useState<"auroc" | "recall" | "false_rejection_rate" | "decision_flip_rate">("auroc");
  const r = q.data?.result;
  const chart = useMemo(() => {
    if (!r) return [];
    const sev = [0, ...r.severities];
    return sev.map((s) => {
      const row: Record<string, number | null> = { severity: s };
      Object.keys(r.robustness_by_degradation).forEach((d) => {
        row[d] = s === 0 ? (metric === "decision_flip_rate" ? 0 : (r.clean as any)[metric] ?? null) : (r.rows.find((x) => x.degradation === d && x.severity === s)?.[metric] ?? null);
      });
      return row;
    });
  }, [r, metric]);
  if (!q.data || !r) return <Spinner />;
  const degs = Object.keys(r.robustness_by_degradation);
  return (
    <div className="space-y-4">
      <p className="text-sm" data-testid="robustness-run-header"><span className="font-semibold">Run #{q.data.id}</span> · model <span className="font-medium">{r.model}</span> ({r.algorithm}) · dataset {r.dataset} · {r.n} {r.split} images · {r.rows.length} conditions · {num(r.seconds, 0)} s · {dt(q.data.created_at)}</p>
      {r.synthetic_data && <Note tone="warn">Synthetic demo data — demonstrates the benchmark only.</Note>}
      <div className="grid gap-4 md:grid-cols-4">
        <div className="card p-4"><p className="muted text-xs uppercase">Robustness score</p><p className="text-2xl font-semibold">{num(r.robustness_score, 2)}</p><p className="muted text-[11px]">mean degraded AUROC ÷ clean AUROC</p></div>
        <div className="card p-4"><p className="muted text-xs uppercase">Clean AUROC</p><p className="text-2xl font-semibold">{num(r.clean.auroc)}</p><p className="muted text-[11px]">n = {r.n} {r.split} images</p></div>
        <div className="card p-4"><p className="muted text-xs uppercase">Clean recall / FRR</p><p className="text-2xl font-semibold">{pct(r.clean.recall, 0)} / {pct(r.clean.false_rejection_rate, 0)}</p></div>
        <div className="card p-4"><p className="muted text-xs uppercase">Worst condition</p><p className="text-lg font-semibold">{r.worst_conditions[0] ? `${titleCase(r.worst_conditions[0].degradation)} s${r.worst_conditions[0].severity}` : "—"}</p><p className="muted text-[11px]">AUROC {num(r.worst_conditions[0]?.auroc)}</p></div>
      </div>
      <Card title="Metric vs. severity" actions={<select className="input w-48 py-1 text-xs" value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} aria-label="Metric"><option value="auroc">Image AUROC</option><option value="recall">Recall @ fail threshold</option><option value="false_rejection_rate">False rejection rate</option><option value="decision_flip_rate">Decision flip rate</option></select>}>
        <div className="h-72"><ResponsiveContainer><LineChart data={chart}><CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis dataKey="severity" fontSize={11} label={{ value: "severity (0 = clean)", position: "insideBottom", offset: -2, fontSize: 11 }} /><YAxis domain={[0, 1]} fontSize={11} /><Tooltip /><Legend wrapperStyle={{ fontSize: 11 }} />
          {degs.map((d, i) => <Line key={d} dataKey={d} stroke={PALETTE[i % PALETTE.length]} strokeWidth={2} dot />)}</LineChart></ResponsiveContainer></div>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Per-degradation robustness" bodyClass="p-0" className="min-w-0">
          <table className="w-full text-sm"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Degradation</th><th className="th text-right">Score</th>{r.severities.map((s) => <th key={s} className="th text-right">AUROC s{s}</th>)}</tr></thead>
            <tbody>{degs.map((d) => (
              <tr key={d} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td">{titleCase(d)}</td><td className="td text-right font-semibold">{num(r.robustness_by_degradation[d], 2)}</td>
                {r.severities.map((s) => { const v = r.rows.find((x) => x.degradation === d && x.severity === s); return <td key={s} className={`td text-right tabular-nums ${v?.auroc != null && v.auroc < 0.7 ? "text-red-600" : ""}`}>{num(v?.auroc ?? null)}</td>; })}</tr>
            ))}</tbody></table>
        </Card>
        <Card title={`Decision flips (${r.decision_flips.length})`} subtitle="Images whose PASS/REVIEW/FAIL decision changed under degradation" bodyClass="p-0" className="min-w-0">
          <div className="max-h-80 overflow-auto"><table className="w-full text-xs"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Image</th><th className="th">Label</th><th className="th">Condition</th><th className="th text-right">Clean → degraded score</th></tr></thead>
            <tbody>{r.decision_flips.slice(0, 100).map((f, i) => <tr key={i} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td font-mono">{f.path.split("/").pop()}</td><td className="td"><Badge tone={f.label === "anomaly" ? "red" : "green"}>{f.label}</Badge></td><td className="td">{titleCase(f.degradation)} s{f.severity}</td><td className="td text-right tabular-nums">{num(f.clean_score)} → {num(f.degraded_score)}</td></tr>)}</tbody></table></div>
        </Card>
      </div>
      {q.data.artifacts.length > 0 && <Card title="Plots"><div className="grid gap-3 md:grid-cols-2">{q.data.artifacts.map((a) => <img key={a} src={a} alt={a.split("/").pop()} className="rounded border border-slate-200 bg-white dark:border-slate-800" />)}</div></Card>}
    </div>
  );
}

function ShiftView({ r }: { r: Record<string, any> }) {
  const rows = [["Clean data, validation threshold", r.clean], ["Shifted data, original threshold", r.shifted_original_threshold], ["Shifted data, recalibrated threshold", r.shifted_recalibrated_threshold]] as const;
  return (
    <div className="space-y-3 text-sm">
      <p><b>Shift:</b> {r.shift}</p>
      <table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Condition</th><th className="th text-right">AUROC</th><th className="th text-right">Recall</th><th className="th text-right">False reject</th><th className="th text-right">F1</th></tr></thead>
        <tbody>{rows.map(([k, v]) => <tr key={k} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td">{k}</td><td className="td text-right">{num(v?.auroc)}</td><td className="td text-right">{pct(v?.recall)}</td><td className="td text-right">{pct(v?.false_rejection_rate)}</td><td className="td text-right">{num(v?.f1)}</td></tr>)}</tbody></table>
      <KV items={[["Fail threshold", `${num(r.original_fail_threshold)} → ${num(r.recalibrated_fail_threshold)} (recalibrated on ${r.n_recalibration_normals} shifted normal images)`], ["OOD flag rate", `clean ${pct(r.ood_flag_rate_clean)} → shifted ${pct(r.ood_flag_rate_shifted)}`], ["Test images", r.n_test]]} />
      <Note>Interpretation: ranking quality (AUROC) shows whether the model can still separate defects after the shift; the threshold rows show how a fixed threshold breaks and how much recalibrating on a few normal images from the new condition recovers. The OOD detector should flag the shifted images for review.</Note>
    </div>
  );
}

export default function Robustness() {
  const { can } = useAuth();
  const isAdmin = can("admin");
  const qc = useQueryClient();
  const [product, setProduct] = useState("");
  const [modelId, setModelId] = useState("");
  const [tab, setTab] = useState<"bench" | "preview" | "shift">("bench");
  const [chosen, setChosen] = useState<string[]>([]);
  const [sev, setSev] = useState<number[]>([1, 3, 5]);
  const [maxSamples, setMaxSamples] = useState(60);
  const [job, setJob] = useState<string | null>(null);
  const [view, setView] = useState<number | null>(null);
  const [pDeg, setPDeg] = useState("motion_blur");
  const [pSev, setPSev] = useState(3);
  const [pSample, setPSample] = useState(0);
  const models = useModels(product);
  const degs = useQuery({ queryKey: ["degradations"], queryFn: () => api<{ degradations: string[]; groups: Record<string, string[]> }>("/api/robustness/degradations") });
  const runs = useQuery({ queryKey: ["evaluations", "robustness", product], queryFn: () => api<EvalRun[]>(`/api/evaluations${qs({ product_key: product })}`) });
  const j = useJob(job);
  useEffect(() => { if (degs.data && !chosen.length) setChosen(["brightness", "uneven_lighting", "gaussian_noise", "motion_blur", "rotation", "jpeg"].filter((d) => degs.data!.degradations.includes(d))); }, [degs.data, chosen.length]);
  useEffect(() => { const ms = models.data ?? []; if (ms.length && !ms.find((m) => String(m.id) === modelId)) setModelId(String((ms.find((m) => m.active_for.length) ?? ms[0]).id)); }, [models.data, modelId]);
  useEffect(() => { if (j.data?.status === "succeeded") { qc.invalidateQueries({ queryKey: ["evaluations"] }); if (j.data.result?.evaluation_id) setView(j.data.result.evaluation_id); } }, [j.data, qc]);
  const run = useMutation({ mutationFn: () => api<Job>("/api/robustness", { body: { model_id: Number(modelId), degradations: chosen, severities: sev, max_samples: maxSamples } }), onSuccess: (x) => setJob(x.id) });
  const shift = useMutation({ mutationFn: () => api<Job>("/api/domain-shift", { body: { model_id: Number(modelId), max_samples: maxSamples } }), onSuccess: (x) => setJob(x.id) });
  const preview = useQuery({
    queryKey: ["rob-preview", modelId, pDeg, pSev, pSample],
    queryFn: () => api<{ sample_path: string; label: string; defect_type: string; n_test: number; clean: { image: string; score: number; decision: string }; degraded: { image: string; score: number; decision: string }; fail_threshold: number; pass_threshold: number }>(`/api/robustness/preview${qs({ model_id: modelId, degradation: pDeg, severity: pSev, sample: pSample })}`),
    enabled: tab === "preview" && !!modelId && isAdmin,
  });
  const robRuns = (runs.data ?? []).filter((r) => r.kind === "robustness");
  const shiftRuns = (runs.data ?? []).filter((r) => r.kind === "domain_shift");
  const shiftDetail = useQuery({ queryKey: ["evaluation", shiftRuns[0]?.id], queryFn: () => api<EvalRun & { result: Record<string, any> }>(`/api/evaluations/${shiftRuns[0].id}`), enabled: tab === "shift" && !!shiftRuns[0] });

  return (
    <>
      <PageHeader title="Robustness lab" description="Stress-test a model against controlled image degradations (lighting, noise, blur, geometry, compression, colour, camera faults, occlusion) at 5 severity levels, on the held-out test split. Also simulates a domain shift with threshold recalibration."
        actions={<>
          <ProductSelect value={product} onChange={setProduct} className="input w-56" />
          <select className="input w-64" value={modelId} onChange={(e) => setModelId(e.target.value)} aria-label="Model">{(models.data ?? []).map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version} ({m.status}{m.active_for.length ? ", active" : ""})</option>)}</select>
        </>} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "bench", label: "Benchmark" }, { id: "preview", label: "Degradation preview" }, { id: "shift", label: "Domain shift" }]} />
      {job && <div className="mb-4"><JobStatus jobId={job} onCancel={() => api(`/api/jobs/${job}/cancel`, { method: "POST" })} /></div>}
      {tab === "bench" && (
        <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
          <div className="space-y-4">
            {isAdmin && (
              <Card title="Run benchmark">
                <div className="space-y-3">
                  {degs.data && Object.entries(degs.data.groups).map(([g, list]) => (
                    <fieldset key={g}><legend className="label">{titleCase(g)}</legend>
                      <div className="flex flex-wrap gap-1.5">{list.map((d) => (
                        <label key={d} className={`cursor-pointer rounded-full border px-2.5 py-1 text-xs ${chosen.includes(d) ? "border-brand-600 bg-brand-700 text-white" : "border-slate-300 dark:border-slate-700"}`}>
                          <input type="checkbox" className="sr-only" checked={chosen.includes(d)} onChange={(e) => setChosen((c) => e.target.checked ? [...c, d] : c.filter((x) => x !== d))} />{titleCase(d)}
                        </label>
                      ))}</div>
                    </fieldset>
                  ))}
                  <fieldset><legend className="label">Severity levels</legend><div className="flex gap-2">{[1, 2, 3, 4, 5].map((s) => <label key={s} className="flex items-center gap-1 text-sm"><input type="checkbox" checked={sev.includes(s)} onChange={(e) => setSev((x) => (e.target.checked ? [...x, s] : x.filter((y) => y !== s)).sort())} />{s}</label>)}</div></fieldset>
                  <Field label="Max test images (balanced subsample)">{(id) => <input id={id} type="number" min={10} max={500} className="input" value={maxSamples} onChange={(e) => setMaxSamples(Number(e.target.value))} />}</Field>
                  <p className="muted text-xs">{chosen.length} degradations × {sev.length} severities × ≤{maxSamples} images = up to {(chosen.length * sev.length * maxSamples).toLocaleString()} inferences.</p>
                  <Button variant="primary" className="w-full" disabled={!modelId || !chosen.length || !sev.length} loading={run.isPending} onClick={() => run.mutate()}><Play className="size-4" /> Run benchmark</Button>
                  <ErrorBox error={run.error} />
                </div>
              </Card>
            )}
            <Card title="Previous runs" bodyClass="p-0">
              {!robRuns.length ? <p className="muted p-4 text-sm">No robustness runs for this product.</p> : <ul>{robRuns.map((r) => (
                <li key={r.id}><button onClick={() => setView(r.id)} className={`flex w-full items-center gap-2 border-b border-slate-100 px-4 py-2 text-left text-sm hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/40 ${view === r.id ? "bg-brand-50 dark:bg-brand-800/20" : ""}`}>
                  <span className="flex-1 truncate" title={`#${r.id} ${r.model ?? ""}`}>#{r.id} {r.model}</span><Badge tone="teal">{num(r.robustness_score ?? null, 2)}</Badge><span className="muted text-xs">{dt(r.created_at)}</span>
                </button></li>
              ))}</ul>}
            </Card>
          </div>
          <div className="min-w-0">{view ? <RobustnessView id={view} /> : <div className="card"><Empty title="Select or run a benchmark" icon={<Shuffle className="size-8" />} /></div>}</div>
        </div>
      )}
      {tab === "preview" && (
        <Card title="See what a degradation does to an image and to the model's decision">
          {!isAdmin ? <Note>Administrator role required.</Note> : (
            <>
              <div className="mb-4 flex flex-wrap items-end gap-3">
                <Field label="Degradation">{(id) => <select id={id} className="input w-52" value={pDeg} onChange={(e) => setPDeg(e.target.value)}>{(degs.data?.degradations ?? []).map((d) => <option key={d} value={d}>{titleCase(d)}</option>)}</select>}</Field>
                <Field label={`Severity ${pSev}`}>{(id) => <input id={id} type="range" min={1} max={5} value={pSev} onChange={(e) => setPSev(Number(e.target.value))} className="w-40" />}</Field>
                <Field label="Test sample #">{(id) => <input id={id} type="number" min={0} className="input w-24" value={pSample} onChange={(e) => setPSample(Number(e.target.value))} />}</Field>
                <Button onClick={() => setPSample((s) => s + 1)}><Eye className="size-4" /> Next sample</Button>
              </div>
              <ErrorBox error={preview.error} />
              {preview.isFetching && <Spinner label="Running model" />}
              {preview.data && (
                <div className="grid gap-4 md:grid-cols-2">
                  {(["clean", "degraded"] as const).map((k) => (
                    <figure key={k} className="space-y-2">
                      <img src={preview.data[k].image} alt={`${k} sample`} className="w-full rounded-lg" />
                      <figcaption className="flex items-center gap-2 text-sm"><b className="capitalize">{k}</b><DecisionBadge decision={preview.data[k].decision} /><span className="muted">score {num(preview.data[k].score)}</span></figcaption>
                    </figure>
                  ))}
                  <p className="muted text-xs md:col-span-2">Sample {preview.data.sample_path} · ground truth: {preview.data.label}{preview.data.defect_type ? ` (${preview.data.defect_type})` : ""} · {preview.data.n_test} test images · raw-threshold decision (PASS &lt; {num(preview.data.pass_threshold)}, FAIL ≥ {num(preview.data.fail_threshold)}).</p>
                </div>
              )}
            </>
          )}
        </Card>
      )}
      {tab === "shift" && (
        <div className="grid gap-5 lg:grid-cols-[320px_1fr]">
          <Card title="Domain-shift experiment">
            <p className="muted mb-3 text-sm">Simulates a lighting change (warmer colour temperature, −20% brightness, gradient) on the test split, measures the damage with the original threshold, then recalibrates the threshold on a small set of shifted normal images from the validation split.</p>
            {isAdmin && <Button variant="primary" disabled={!modelId} loading={shift.isPending} onClick={() => shift.mutate()}><Play className="size-4" /> Run experiment</Button>}
            <ErrorBox error={shift.error} />
          </Card>
          <Card title={shiftRuns[0] ? `Latest result (#${shiftRuns[0].id}, ${shiftRuns[0].model ?? ""})` : "Result"}>
            {shiftDetail.data?.result ? <ShiftView r={shiftDetail.data.result} /> : <p className="muted text-sm">No experiment run for this product yet.</p>}
          </Card>
        </div>
      )}
    </>
  );
}
