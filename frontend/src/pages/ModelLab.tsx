import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, GitCompare, Play, Undo2, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { JobStatus, useJob } from "../components/inspection";
import ProductSelect from "../components/ProductSelect";
import { Badge, Button, Card, ConfirmButton, Empty, ErrorBox, Field, Modal, Note, PageHeader, Spinner, StatusBadge } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ci, dt, ms, num, pct } from "../lib/format";
import { useDatasets, useModels, useProducts } from "../lib/hooks";
import { useToast } from "../lib/toast";
import type { Job } from "../lib/types";

interface EvalRow { model: string; algorithm: string; dataset: string; n: number; auroc: number | null; auroc_ci95?: number[]; ap: number | null; f1: number | null; precision: number | null; recall: number | null; balanced_accuracy: number | null; miss_rate: number | null; false_rejection_rate: number | null; "recall_at_fpr_0.05": number | null; auto_pass_miss_rate: number | null; review_rate: number | null; pixel_auroc: number | null; aupro: number | null; dice: number | null; ece: number | null; latency_ms: number | null; throughput: number | null }
interface EvalRun { id: number; kind: string; product_key: string; dataset_name: string; model_ids: number[]; created_at: string; job_id?: string; robustness_score?: number | null; model?: string | null }
interface EvalDetail extends EvalRun { result: { rows?: EvalRow[]; results?: { synthetic_data?: boolean; split?: string; n_anomalous?: number; n_normal?: number }[] } & Record<string, any>; artifacts: string[] }

export function EvalTable({ rows }: { rows: EvalRow[] }) {
  const best = (k: keyof EvalRow, hi = true) => {
    const v = rows.map((r) => r[k] as number | null).filter((x): x is number => typeof x === "number");
    return v.length ? (hi ? Math.max(...v) : Math.min(...v)) : null;
  };
  const cell = (r: EvalRow, k: keyof EvalRow, f: (v: number | null) => string, hi = true) => {
    const v = r[k] as number | null;
    return <td className={`td text-right tabular-nums ${v !== null && v === best(k, hi) && rows.length > 1 ? "font-bold text-brand-700 dark:text-brand-500" : ""}`}>{f(v)}</td>;
  };
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead><tr className="border-b border-slate-200 dark:border-slate-800">
          {["Model", "n", "AUROC [95% CI]", "AP", "F1", "Recall", "Miss rate", "False reject", "Recall@5%FPR", "Auto-PASS miss", "Review rate", "Pixel AUROC", "AUPRO", "Dice", "ECE", "Latency"].map((h) => <th key={h} className="th text-right first:text-left">{h}</th>)}
        </tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.model} className="border-b border-slate-100 dark:border-slate-800/60">
            <td className="td font-medium whitespace-nowrap">{r.model}<p className="muted text-xs">{r.algorithm}</p></td>
            <td className="td text-right">{r.n}</td>
            <td className={`td text-right whitespace-nowrap tabular-nums ${r.auroc === best("auroc") && rows.length > 1 ? "font-bold text-brand-700 dark:text-brand-500" : ""}`}>{num(r.auroc)} <span className="muted text-[10px]">{ci(r.auroc_ci95)}</span></td>
            {cell(r, "ap", (v) => num(v))}{cell(r, "f1", (v) => num(v))}{cell(r, "recall", (v) => pct(v))}{cell(r, "miss_rate", (v) => pct(v), false)}{cell(r, "false_rejection_rate", (v) => pct(v), false)}
            {cell(r, "recall_at_fpr_0.05", (v) => pct(v))}{cell(r, "auto_pass_miss_rate", (v) => pct(v), false)}{cell(r, "review_rate", (v) => pct(v), false)}{cell(r, "pixel_auroc", (v) => num(v))}{cell(r, "aupro", (v) => num(v))}{cell(r, "dice", (v) => num(v))}{cell(r, "ece", (v) => num(v), false)}{cell(r, "latency_ms", (v) => ms(v), false)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function EvaluationModal({ id, onClose }: { id: number | null; onClose: () => void }) {
  const q = useQuery({ queryKey: ["evaluation", id], queryFn: () => api<EvalDetail>(`/api/evaluations/${id}`), enabled: !!id });
  const synth = q.data?.result.results?.some((r) => r.synthetic_data);
  return (
    <Modal open={!!id} onClose={onClose} title={`Evaluation #${id}`} wide>
      {!q.data ? <Spinner /> : (
        <div className="space-y-3">
          <p className="muted text-sm">{q.data.kind} · dataset {q.data.dataset_name} · split {q.data.result.results?.[0]?.split ?? "test"} · {dt(q.data.created_at)}</p>
          {synth && <Note tone="warn">Synthetic demo data — these numbers demonstrate the pipeline and are not evidence of real-world performance.</Note>}
          {q.data.result.rows && <EvalTable rows={q.data.result.rows} />}
          <div className="grid gap-3 md:grid-cols-2">{q.data.artifacts.map((a) => <img key={a} src={a} alt={a.split("/").pop()} className="rounded border border-slate-200 bg-white dark:border-slate-800" />)}</div>
        </div>
      )}
    </Modal>
  );
}

export default function ModelLab() {
  const [product, setProduct] = useState("");
  const [selected, setSelected] = useState<number[]>([]);
  const [evalDataset, setEvalDataset] = useState("");
  const [split, setSplit] = useState<"test" | "val">("test");
  const [evalJob, setEvalJob] = useState<string | null>(null);
  const [trainOpen, setTrainOpen] = useState(false);
  const [viewEval, setViewEval] = useState<number | null>(null);
  const { can } = useAuth();
  const isAdmin = can("admin");
  const toast = useToast();
  const qc = useQueryClient();
  const products = useProducts();
  const models = useModels(product);
  const datasets = useDatasets();
  const evals = useQuery({ queryKey: ["evaluations", product], queryFn: () => api<EvalRun[]>(`/api/evaluations${qs({ product_key: product, kind: "benchmark" })}`) });
  const acts = useQuery({ queryKey: ["activations", product], queryFn: () => api<{ id: number; model_id: number | null; previous_model_id: number | null; reason: string; actor: string | null; created_at: string }[]>(`/api/products/${product}/activations`), enabled: !!product });
  const evalJobQ = useJob(evalJob);
  const p = products.data?.find((x) => x.key === product);
  const dsForProduct = (datasets.data ?? []).filter((d) => d.product_key === product);
  useEffect(() => { setSelected([]); setEvalDataset(""); }, [product]);
  useEffect(() => { if (!evalDataset && dsForProduct.length) setEvalDataset(String(dsForProduct[0].id)); }, [dsForProduct, evalDataset]);
  useEffect(() => {
    if (evalJobQ.data?.status === "succeeded") { qc.invalidateQueries({ queryKey: ["evaluations"] }); if (evalJobQ.data.result?.evaluation_id) setViewEval(evalJobQ.data.result.evaluation_id); setEvalJob(null); }
  }, [evalJobQ.data, qc]);

  const invalidate = () => { qc.invalidateQueries({ queryKey: ["models"] }); qc.invalidateQueries({ queryKey: ["products"] }); qc.invalidateQueries({ queryKey: ["activations"] }); };
  const activate = useMutation({ mutationFn: (id: number) => api(`/api/products/${product}/activate`, { body: { model_id: id, reason: "activated from Model Lab" } }), onSuccess: () => { invalidate(); toast("ok", "Model activated"); }, onError: (e: Error) => toast("error", e.message) });
  const rollback = useMutation({ mutationFn: () => api(`/api/products/${product}/rollback`, { method: "POST" }), onSuccess: () => { invalidate(); toast("ok", "Rolled back to previous model"); }, onError: (e: Error) => toast("error", e.message) });
  const evaluate = useMutation({
    mutationFn: () => api<Job>("/api/models/evaluate", { body: { dataset_id: Number(evalDataset), model_ids: selected, split } }),
    onSuccess: (j) => setEvalJob(j.id),
  });

  return (
    <>
      <PageHeader title="Model lab" description="Compare model versions on the same held-out data, approve candidates and control which approved model is active per product. Training never activates a model automatically."
        actions={<>
          <ProductSelect value={product} onChange={setProduct} className="input w-60" />
          {isAdmin && <Button variant="primary" onClick={() => setTrainOpen(true)} disabled={!product}><FlaskConical className="size-4" /> Train new model</Button>}
        </>} />
      {p && (
        <Card className="mb-5" title={`${p.name}${p.synthetic ? " (synthetic demo)" : ""}`} subtitle={p.description}
          actions={isAdmin && <ConfirmButton size="sm" onConfirm={() => rollback.mutate()} confirmText="Re-activate the model that was active before the current one?" disabled={!acts.data?.length}><Undo2 className="size-3.5" /> Roll back</ConfirmButton>}>
          <div className="flex flex-wrap items-center gap-4 text-sm">
            <span>Active model: {p.active_model ? <Link className="font-semibold underline" to={`/models/${p.active_model.id}`}>{p.active_model.name} v{p.active_model.version}</Link> : <Badge tone="amber">none</Badge>}</span>
            {p.secondary_model_id && <span>Second opinion: model #{p.secondary_model_id}</span>}
            {p.active_model && <span className="muted">thresholds PASS &lt; {num(p.active_model.thresholds.pass_threshold)} · FAIL ≥ {num(p.active_model.thresholds.fail_threshold)}</span>}
          </div>
          {acts.data && acts.data.length > 0 && (
            <details className="mt-3 text-xs"><summary className="muted cursor-pointer">Activation history ({acts.data.length})</summary>
              <ul className="mt-2 space-y-1">{acts.data.map((a) => <li key={a.id}>{dt(a.created_at)} — model #{a.model_id ?? "none"} (previous #{a.previous_model_id ?? "none"}) by {a.actor ?? "system"}: {a.reason}</li>)}</ul>
            </details>
          )}
        </Card>
      )}
      <ErrorBox error={models.error} />
      <Card title="Model versions" bodyClass="p-0" actions={<span className="muted text-xs">Metrics: held-out test split (report-only; never used for threshold selection)</span>}>
        {models.isLoading ? <Spinner /> : !models.data?.length ? <Empty title="No models for this product" icon={<FlaskConical className="size-8" />}>Train a model from a registered dataset.</Empty> : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th w-8"><span className="sr-only">Select</span></th><th className="th">Model</th><th className="th">Status</th><th className="th text-right">Val AUROC</th><th className="th text-right">Test AUROC</th><th className="th text-right">Test F1</th><th className="th text-right">Miss rate</th><th className="th text-right">False reject</th><th className="th text-right">Pixel AUROC</th><th className="th text-right">Latency</th><th className="th">Created</th><th className="th"></th></tr></thead>
              <tbody>{models.data.map((m) => (
                <tr key={m.id} className="border-b border-slate-100 dark:border-slate-800/60">
                  <td className="td"><input type="checkbox" aria-label={`Select ${m.name} for comparison`} checked={selected.includes(m.id)} onChange={(e) => setSelected((s) => e.target.checked ? [...s, m.id] : s.filter((x) => x !== m.id))} /></td>
                  <td className="td"><Link to={`/models/${m.id}`} className="font-medium hover:underline">{m.name}</Link><p className="muted text-xs">v{m.version} · {m.algorithm}{m.has_classifier ? " · +classifier" : ""}</p></td>
                  <td className="td"><div className="flex flex-wrap gap-1"><StatusBadge status={m.status} />{m.active_for.length > 0 && <Badge tone="teal"><Zap className="size-3" />active</Badge>}</div></td>
                  <td className="td text-right tabular-nums">{num(m.summary.val_auroc)}</td>
                  <td className="td text-right font-semibold tabular-nums">{num(m.summary.test_auroc)}</td>
                  <td className="td text-right tabular-nums">{num(m.summary.test_f1)}</td>
                  <td className="td text-right tabular-nums">{pct(m.summary.test_miss_rate)}</td>
                  <td className="td text-right tabular-nums">{pct(m.summary.test_frr)}</td>
                  <td className="td text-right tabular-nums">{num(m.summary.pixel_auroc)}</td>
                  <td className="td text-right tabular-nums">{ms(m.summary.latency_ms)}</td>
                  <td className="td muted text-xs whitespace-nowrap">{dt(m.created_at)}</td>
                  <td className="td">{isAdmin && m.status === "approved" && m.active_for.length === 0 && <Button size="sm" onClick={() => activate.mutate(m.id)} loading={activate.isPending && activate.variables === m.id}>Activate</Button>}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>
      {isAdmin && (
        <Card className="mt-5" title="Compare on the same data" subtitle="Runs a fresh evaluation of the selected models on one dataset split">
          <div className="flex flex-wrap items-end gap-3">
            <Field label="Dataset">{(id) => <select id={id} className="input w-56" value={evalDataset} onChange={(e) => setEvalDataset(e.target.value)}>{dsForProduct.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.version})</option>)}</select>}</Field>
            <Field label="Split">{(id) => <select id={id} className="input w-44" value={split} onChange={(e) => setSplit(e.target.value as "test" | "val")}><option value="test">held-out test</option><option value="val">validation</option></select>}</Field>
            <Button variant="primary" disabled={selected.length < 1 || !evalDataset || !!evalJob} loading={evaluate.isPending} onClick={() => evaluate.mutate()}><GitCompare className="size-4" /> Evaluate {selected.length} model(s)</Button>
          </div>
          <ErrorBox error={evaluate.error} />
          {evalJob && <div className="mt-3"><JobStatus jobId={evalJob} /></div>}
        </Card>
      )}
      <Card className="mt-5" title="Evaluation runs" bodyClass="p-0">
        {!evals.data?.length ? <p className="muted p-4 text-sm">No comparison runs yet.</p> : (
          <ul>{evals.data.map((e) => (
            <li key={e.id}><button onClick={() => setViewEval(e.id)} className="flex w-full items-center gap-3 border-b border-slate-100 px-4 py-2 text-left text-sm hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/40">
              <span className="font-medium">#{e.id}</span><span className="flex-1">{e.dataset_name} · {e.model_ids.length} model(s)</span><span className="muted text-xs">{dt(e.created_at)}</span>
            </button></li>
          ))}</ul>
        )}
      </Card>
      <EvaluationModal id={viewEval} onClose={() => setViewEval(null)} />
      <TrainModal open={trainOpen} onClose={() => setTrainOpen(false)} product={product} />
    </>
  );
}

function TrainModal({ open, onClose, product }: { open: boolean; onClose: () => void; product: string }) {
  const datasets = useDatasets();
  const algos = useQuery({ queryKey: ["algorithms"], queryFn: () => api<{ key: string; description: string }[]>("/api/algorithms") });
  const ds = (datasets.data ?? []).filter((d) => d.product_key === product);
  const [dataset, setDataset] = useState("");
  const [algo, setAlgo] = useState("patchcore");
  const [name, setName] = useState("");
  const [maxImgs, setMaxImgs] = useState("");
  const [recall, setRecall] = useState(0.98);
  const [notes, setNotes] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const qc = useQueryClient();
  const j = useJob(job);
  useEffect(() => { if (open && !dataset && ds.length) setDataset(String(ds[0].id)); }, [open, ds, dataset]);
  useEffect(() => { if (j.data?.status === "succeeded") qc.invalidateQueries({ queryKey: ["models"] }); }, [j.data, qc]);
  const train = useMutation({
    mutationFn: () => api<Job>("/api/models/train", { body: { dataset_id: Number(dataset), algorithm: algo, name: name || null, max_train_images: maxImgs ? Number(maxImgs) : null, target_recall: recall, config: {}, notes } }),
    onSuccess: (x) => setJob(x.id),
  });
  const cancel = useMutation({ mutationFn: () => api(`/api/jobs/${job}/cancel`, { method: "POST" }) });
  return (
    <Modal open={open} onClose={onClose} title="Train a candidate model">
      <div className="space-y-3">
        <Note>Training uses only normal images of the dataset's <b>train</b> split. Thresholds and calibration are selected on the <b>validation</b> split; the held-out <b>test</b> split is evaluated once for reporting. The new model is registered as a <b>candidate</b> and must be approved and activated by an administrator.</Note>
        <Field label="Dataset">{(id) => <select id={id} className="input" value={dataset} onChange={(e) => setDataset(e.target.value)}>{ds.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.sample_count} samples{d.synthetic ? " · synthetic" : ""}</option>)}</select>}</Field>
        <Field label="Algorithm" hint={algos.data?.find((a) => a.key === algo)?.description}>{(id) => <select id={id} className="input" value={algo} onChange={(e) => setAlgo(e.target.value)}>{(algos.data ?? []).map((a) => <option key={a.key} value={a.key}>{a.description.split(" - ")[0]}</option>)}</select>}</Field>
        <div className="grid grid-cols-2 gap-2">
          <Field label="Name (optional)">{(id) => <input id={id} className="input" value={name} onChange={(e) => setName(e.target.value)} pattern="[a-z0-9][a-z0-9_-]*" />}</Field>
          <Field label="Max training images" hint="Limit for low-RAM machines">{(id) => <input id={id} type="number" min={2} className="input" value={maxImgs} onChange={(e) => setMaxImgs(e.target.value)} placeholder="all" />}</Field>
        </div>
        <Field label={`Target recall for PASS threshold: ${recall.toFixed(2)}`}>{(id) => <input id={id} type="range" min={0.8} max={1} step={0.01} value={recall} onChange={(e) => setRecall(Number(e.target.value))} className="w-full" />}</Field>
        <Field label="Notes">{(id) => <input id={id} className="input" value={notes} onChange={(e) => setNotes(e.target.value)} />}</Field>
        <ErrorBox error={train.error} />
        {job ? <JobStatus jobId={job} onCancel={() => cancel.mutate()} /> : <Button variant="primary" className="w-full" disabled={!dataset} loading={train.isPending} onClick={() => train.mutate()}><Play className="size-4" /> Start training</Button>}
        {j.data?.status === "succeeded" && j.data.result?.model_id && <Link className="block text-center text-sm underline" to={`/models/${j.data.result.model_id}`} onClick={onClose}>Open the new candidate model →</Link>}
      </div>
    </Modal>
  );
}
