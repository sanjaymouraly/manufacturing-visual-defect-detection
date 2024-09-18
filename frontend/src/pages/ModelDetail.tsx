import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Brain, CheckCircle2, Download, FileCode2, RotateCcw, Save, XCircle, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { JobStatus } from "../components/inspection";
import { Badge, Button, Card, ConfirmButton, ErrorBox, Field, KV, Note, PageHeader, Spinner, StatusBadge, Tabs } from "../components/ui";
import { api, download } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ci, dt, ms, num, pct, titleCase } from "../lib/format";
import { useToast } from "../lib/toast";
import type { Job, ModelVersion } from "../lib/types";

function MetricsBlock({ m, which }: { m: Record<string, any>; which: "test" | "val" }) {
  if (!m || !m.ranking) return <p className="muted text-sm">{which === "test" ? "No held-out test evaluation recorded for this model." : "No validation metrics."}</p>;
  const a = m.at_fail_threshold ?? {};
  const c = a.confusion ?? {};
  const pol = m.policy;
  const roc = m.ranking.roc_curve ? m.ranking.roc_curve.fpr.map((f: number, i: number) => ({ fpr: f, tpr: m.ranking.roc_curve.tpr[i] })) : null;
  return (
    <div className="space-y-4">
      {m.synthetic_data && <Note tone="warn">Synthetic demo data — not evidence of real-world performance.</Note>}
      <div className="grid gap-4 lg:grid-cols-3">
        <div>
          <h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">Ranking (threshold-free)</h3>
          <KV items={[["n (anomalous / normal)", `${m.n} (${m.n_anomalous ?? a.n_anomalous ?? "?"} / ${m.n_normal ?? a.n_normal ?? "?"})`], ["Image AUROC", <>{num(m.ranking.auroc)} <span className="muted text-xs">{ci(m.auroc_ci95)}</span></>], ["Average precision", num(m.ranking.average_precision)],
            ...Object.entries(m.ranking.recall_at_fpr ?? {}).map(([k, v]) => [`Recall @ ${pct(Number(k), 0)} FPR`, pct(v as number)] as [string, string])]} />
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">At reject threshold ({num(a.threshold)})</h3>
          <KV items={[["Precision", num(a.precision)], ["Recall", <>{pct(a.recall)} <span className="muted text-xs">{ci(a.recall_ci95, 2)}</span></>], ["F1", num(a.f1)], ["Balanced accuracy", num(a.balanced_accuracy)],
            ["Miss rate", pct(a.miss_rate)], ["False rejection rate", <>{pct(a.false_rejection_rate)} <span className="muted text-xs">{ci(a.false_rejection_rate_ci95, 2)}</span></>]]} />
          <table className="mt-3 text-xs"><tbody>
            <tr><td></td><td className="px-2 font-semibold">pred. defect</td><td className="px-2 font-semibold">pred. normal</td></tr>
            <tr><td className="pr-2 font-semibold">defect</td><td className="bg-emerald-100 px-2 text-center dark:bg-emerald-950">{c.tp}</td><td className="bg-red-100 px-2 text-center dark:bg-red-950">{c.fn}</td></tr>
            <tr><td className="pr-2 font-semibold">normal</td><td className="bg-amber-100 px-2 text-center dark:bg-amber-950">{c.fp}</td><td className="bg-emerald-100 px-2 text-center dark:bg-emerald-950">{c.tn}</td></tr>
          </tbody></table>
        </div>
        <div>
          <h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">Three-way policy & runtime</h3>
          {pol ? <KV items={[["Auto-PASS of defects", <>{pct(pol.auto_pass_miss_rate)} <span className="muted text-xs">{ci(pol.auto_pass_miss_rate_ci95, 2)}</span></>], ["Review rate", pct(pol.review_rate)],
            ["Normals → P/R/F", `${pol.normal.PASS}/${pol.normal.REVIEW}/${pol.normal.FAIL}`], ["Defects → P/R/F", `${pol.anomalous.PASS}/${pol.anomalous.REVIEW}/${pol.anomalous.FAIL}`]]} /> : <p className="muted text-sm">—</p>}
          <div className="mt-2"><KV items={[["Latency / image", ms(m.latency_ms_per_image)], ["Throughput", m.throughput_img_per_s ? `${m.throughput_img_per_s.toFixed(1)} img/s` : "—"], ["Peak memory", m.peak_memory_mb ? `${m.peak_memory_mb.toFixed(0)} MB` : "—"]]} /></div>
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {roc && (
          <div><h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">ROC curve</h3>
            <div className="h-56"><ResponsiveContainer><LineChart data={roc}><CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis dataKey="fpr" type="number" domain={[0, 1]} fontSize={11} label={{ value: "FPR", position: "insideBottom", offset: -2, fontSize: 11 }} /><YAxis domain={[0, 1]} fontSize={11} /><Tooltip /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} strokeDasharray="4 4" /><Line dataKey="tpr" type="stepAfter" stroke="#0d9488" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div>
          </div>
        )}
        {m.calibration_test?.reliability && (
          <div><h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">Calibration (ECE {num(m.calibration_test.ece)}, Brier {num(m.calibration_test.brier)})</h3>
            <div className="h-56"><ResponsiveContainer><ScatterChart><CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis dataKey="mean_prob" type="number" domain={[0, 1]} name="predicted" fontSize={11} /><YAxis dataKey="frac_pos" type="number" domain={[0, 1]} name="observed" fontSize={11} /><ZAxis dataKey="count" range={[30, 300]} /><Tooltip /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} strokeDasharray="4 4" /><Scatter data={m.calibration_test.reliability} fill="#6366f1" /></ScatterChart></ResponsiveContainer></div>
          </div>
        )}
        {m.pixel && (
          <div><h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">Localization (pixel level)</h3>
            <KV items={[["Pixel AUROC", num(m.pixel.pixel_auroc)], ["Pixel AP", num(m.pixel.pixel_ap)], ["AUPRO (FPR ≤ 0.3)", num(m.pixel["aupro_0.3"])], ["IoU", num(m.pixel.iou)], ["Dice", num(m.pixel.dice)], ["Region recall", `${pct(m.pixel.region_recall)} of ${m.pixel.n_regions}`], ["Defect pixel fraction", pct(m.pixel.defect_pixel_fraction, 2)]]} />
          </div>
        )}
      </div>
      {m.per_defect_type_recall && (
        <div><h3 className="mb-2 text-xs font-semibold tracking-wide uppercase">Recall per annotated defect type (with 95% Wilson CI)</h3>
          <div className="flex flex-wrap gap-2">{Object.entries(m.per_defect_type_recall as Record<string, { recall: number; detected: number; n: number; ci95: number[] }>).map(([k, v]) => (
            <div key={k} className="rounded-lg border border-slate-200 px-3 py-2 text-xs dark:border-slate-800"><p className="font-medium">{titleCase(k)}</p><p className="tabular-nums">{pct(v.recall)} ({v.detected}/{v.n}) <span className="muted">{ci(v.ci95, 2)}</span></p></div>
          ))}</div>
        </div>
      )}
    </div>
  );
}

export default function ModelDetail() {
  const { id } = useParams();
  const { can } = useAuth();
  const isAdmin = can("admin");
  const toast = useToast();
  const qc = useQueryClient();
  const [tab, setTab] = useState<"test" | "val" | "thresholds" | "lifecycle" | "card">("test");
  const q = useQuery({ queryKey: ["model", id], queryFn: () => api<ModelVersion>(`/api/models/${id}`) });
  const card = useQuery({ queryKey: ["model-card", id], queryFn: () => api<string>(`/api/models/${id}/card`), enabled: tab === "card" });
  const m = q.data;
  const [fail, setFail] = useState("");
  const [pass, setPass] = useState("");
  const [seg, setSeg] = useState("");
  const [reason, setReason] = useState("");
  const [notes, setNotes] = useState("");
  const [job, setJob] = useState<string | null>(null);
  useEffect(() => { if (m) { setFail(String(m.thresholds.fail_threshold)); setPass(String(m.thresholds.pass_threshold)); setSeg(m.thresholds.seg_threshold != null ? String(m.thresholds.seg_threshold) : ""); } }, [m]);
  const refresh = () => { qc.invalidateQueries({ queryKey: ["model", id] }); qc.invalidateQueries({ queryKey: ["models"] }); qc.invalidateQueries({ queryKey: ["products"] }); };
  const onErr = (e: Error) => toast("error", e.message);
  const saveTh = useMutation({ mutationFn: () => api(`/api/models/${id}/thresholds`, { method: "PUT", body: { fail_threshold: Number(fail), pass_threshold: Number(pass), seg_threshold: seg ? Number(seg) : null, reason } }), onSuccess: () => { refresh(); setReason(""); toast("ok", "Thresholds updated (audited)"); }, onError: onErr });
  const resetTh = useMutation({ mutationFn: () => api(`/api/models/${id}/thresholds/reset`, { method: "POST" }), onSuccess: () => { refresh(); toast("ok", "Thresholds reset to validation-selected values"); }, onError: onErr });
  const setStatus = useMutation({ mutationFn: (status: string) => api(`/api/models/${id}/status`, { body: { status, notes } }), onSuccess: () => { refresh(); setNotes(""); toast("ok", "Status updated"); }, onError: onErr });
  const activate = useMutation({ mutationFn: () => api(`/api/products/${m!.product_key}/activate`, { body: { model_id: m!.id, reason: notes || "activated from model page" } }), onSuccess: () => { refresh(); toast("ok", "Model is now active"); }, onError: onErr });
  const startJob = useMutation({ mutationFn: (path: string) => api<Job>(path, { method: "POST" }), onSuccess: (j) => setJob(j.id), onError: onErr });

  if (q.isLoading) return <Spinner />;
  if (q.error || !m) return <ErrorBox error={q.error ?? "Not found"} />;
  const thErr = Number(pass) > Number(fail) ? "PASS threshold must not exceed the FAIL threshold" : null;
  return (
    <>
      <Link to="/models" className="muted mb-2 inline-flex items-center gap-1 text-sm hover:underline"><ArrowLeft className="size-4" /> Model lab</Link>
      <PageHeader title={`${m.name}`} description={<>v{m.version} · {m.algorithm} · product {m.product_key} · trained on {m.dataset?.name ?? "?"} ({m.dataset?.version ?? ""})</>}
        actions={<div className="flex flex-wrap items-center gap-2"><StatusBadge status={m.status} />{m.active_for.length > 0 && <Badge tone="teal"><Zap className="size-3" /> active for {m.active_for.join(", ")}</Badge>}{m.calibrated && <Badge tone="violet">calibrated</Badge>}{m.has_classifier && <Badge tone="blue">defect classifier</Badge>}</div>} />
      {job && <div className="mb-4"><JobStatus jobId={job} /></div>}
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "test", label: "Held-out test" }, { id: "val", label: "Validation" }, { id: "thresholds", label: "Thresholds" }, { id: "lifecycle", label: "Lifecycle & export" }, { id: "card", label: "Model card" }]} />
      {tab === "test" && <Card><MetricsBlock m={m.test_metrics} which="test" /></Card>}
      {tab === "val" && <Card><Note>Validation metrics were used to select thresholds and fit calibration, so they are optimistic. Use held-out test metrics for reporting.</Note><div className="mt-3"><MetricsBlock m={m.validation_metrics} which="val" /></div></Card>}
      {tab === "thresholds" && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card title="Decision thresholds" subtitle={m.original_thresholds.method}>
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2">
                <Field label="PASS below">{(fid) => <input id={fid} type="number" step="any" className="input" value={pass} onChange={(e) => setPass(e.target.value)} disabled={!isAdmin} />}</Field>
                <Field label="FAIL at or above">{(fid) => <input id={fid} type="number" step="any" className="input" value={fail} onChange={(e) => setFail(e.target.value)} disabled={!isAdmin} />}</Field>
                <Field label="Segmentation">{(fid) => <input id={fid} type="number" step="any" className="input" value={seg} onChange={(e) => setSeg(e.target.value)} disabled={!isAdmin} />}</Field>
              </div>
              <p className="muted text-xs">Scores between the two thresholds go to human REVIEW. Validation-selected: PASS &lt; {num(m.original_thresholds.pass_threshold)}, FAIL ≥ {num(m.original_thresholds.fail_threshold)}.</p>
              {thErr && <p className="text-sm text-red-600">{thErr}</p>}
              {isAdmin && (
                <>
                  <Field label="Reason for change (required, audited)">{(fid) => <input id={fid} className="input" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} />}</Field>
                  <div className="flex gap-2">
                    <Button variant="primary" disabled={!reason.trim() || !!thErr} loading={saveTh.isPending} onClick={() => saveTh.mutate()}><Save className="size-4" /> Save thresholds</Button>
                    <ConfirmButton onConfirm={() => resetTh.mutate()} confirmText="Restore the thresholds selected on the validation split?"><RotateCcw className="size-4" /> Reset</ConfirmButton>
                  </div>
                  <Note tone="warn">Never tune thresholds by looking at held-out test results — that invalidates the reported metrics. Use validation data or production review outcomes.</Note>
                </>
              )}
            </div>
          </Card>
          <Card title="Calibration">
            {m.calibration ? <KV items={Object.entries(m.calibration).map(([k, v]) => [k, typeof v === "number" ? num(v, 4) : String(v)])} /> : <p className="muted text-sm">Not calibrated (needs both normal and anomalous validation samples). The UI then shows only raw scores.</p>}
            <p className="muted mt-2 text-xs">Calibrated probability = sigmoid(a·score + b), fitted on the validation split (Platt scaling). It estimates the chance an image with this score is defective under validation-like conditions; it is not reliable for out-of-distribution images.</p>
          </Card>
        </div>
      )}
      {tab === "lifecycle" && (
        <div className="grid gap-5 lg:grid-cols-2">
          <Card title="Approval & activation">
            <KV items={[["Status", <StatusBadge status={m.status} />], ["Created", dt(m.created_at)], ["Approved", dt(m.approved_at)], ["Weights SHA-256", <span className="font-mono text-[11px] break-all">{m.weights_sha256}</span>], ["Artifact", <span className="font-mono text-[11px] break-all">{m.artifact_dir}</span>], ["Notes", m.notes || "—"]]} />
            {isAdmin && (
              <div className="mt-4 space-y-3">
                <Field label="Notes / reason">{(fid) => <input id={fid} className="input" value={notes} onChange={(e) => setNotes(e.target.value)} />}</Field>
                <div className="flex flex-wrap gap-2">
                  {m.status !== "approved" && <ConfirmButton variant="success" onConfirm={() => setStatus.mutate("approved")} confirmText="Approve this model? Only approved models can be activated. Review its held-out test metrics first."><CheckCircle2 className="size-4" /> Approve</ConfirmButton>}
                  {m.status === "candidate" && <Button variant="danger" onClick={() => setStatus.mutate("rejected")}><XCircle className="size-4" /> Reject</Button>}
                  {m.status === "approved" && m.active_for.length === 0 && <Button variant="primary" onClick={() => activate.mutate()} loading={activate.isPending}><Zap className="size-4" /> Activate for {m.product_key}</Button>}
                  {m.status !== "archived" && m.active_for.length === 0 && <Button onClick={() => setStatus.mutate("archived")}>Archive</Button>}
                </div>
              </div>
            )}
          </Card>
          <Card title="Extensions & export">
            <div className="space-y-3 text-sm">
              <div>
                <p className="font-medium">Defect-type classifier</p>
                <p className="muted text-xs">Trains a small classifier on embeddings of annotated defect classes from the training dataset (requires per-class labels; never inferred from masks).</p>
                {m.classifier_metrics && <pre className="mt-1 max-h-40 overflow-auto rounded bg-slate-100 p-2 text-[11px] dark:bg-slate-800">{JSON.stringify(m.classifier_metrics, null, 1)}</pre>}
                {isAdmin && <Button size="sm" className="mt-2" onClick={() => startJob.mutate(`/api/models/${id}/classifier`)}><Brain className="size-3.5" /> {m.has_classifier ? "Retrain" : "Train"} classifier</Button>}
              </div>
              <div>
                <p className="font-medium">ONNX export</p>
                <p className="muted text-xs">Exports the feature extractor + scoring head and verifies numerical parity against PyTorch (PatchCore and PaDiM).</p>
                {isAdmin && <div className="mt-2 flex gap-2">
                  <Button size="sm" onClick={() => startJob.mutate(`/api/models/${id}/export-onnx`)} disabled={m.algorithm === "pixelstats"}><FileCode2 className="size-3.5" /> Export ONNX</Button>
                  <Button size="sm" onClick={() => download(`/api/models/${id}/onnx`, `${m.name}.onnx`).catch((e) => toast("error", e.message))}><Download className="size-3.5" /> Download</Button>
                </div>}
              </div>
              <div>
                <p className="font-medium">Training history</p>
                <ul className="muted mt-1 space-y-0.5 text-xs">{(m.history ?? []).map((h, i) => <li key={i}>{Object.entries(h).map(([k, v]) => `${k}: ${typeof v === "number" ? Math.round(v * 100) / 100 : String(v)}`).join(" · ")}</li>)}</ul>
              </div>
            </div>
          </Card>
        </div>
      )}
      {tab === "card" && <Card>{card.isLoading ? <Spinner /> : <pre className="text-xs leading-relaxed whitespace-pre-wrap">{card.data}</pre>}</Card>}
    </>
  );
}
