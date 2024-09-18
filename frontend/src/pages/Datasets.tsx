import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Database, GitBranchPlus, RefreshCw, ShieldCheck, Upload, XCircle } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { JobStatus } from "../components/inspection";
import { Badge, Button, Card, Empty, ErrorBox, Field, KV, Modal, Note, PageHeader, Spinner, StatusBadge, Tabs } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { dt, int } from "../lib/format";
import { useDatasets } from "../lib/hooks";
import { useToast } from "../lib/toast";
import type { DatasetRec, Job } from "../lib/types";

interface Sample { index: number; path: string; label: string; defect_type: string; split: string; has_mask: boolean; group: string }
interface Feedback { id: number; inspection_id: string; review_id: number; product_key: string; label: string; defect_type: string; status: string; dataset_id: number | null; created_at: string; model_decision: string | null; thumbnail: string | null }

function DatasetDetail({ ds, onClose }: { ds: DatasetRec | null; onClose: () => void }) {
  const [split, setSplit] = useState("");
  const [label, setLabel] = useState("");
  const [showMask, setShowMask] = useState(false);
  const { can } = useAuth();
  const qc = useQueryClient();
  const detail = useQuery({ queryKey: ["dataset", ds?.id], queryFn: () => api<DatasetRec>(`/api/datasets/${ds!.id}`), enabled: !!ds });
  const samples = useQuery({ queryKey: ["samples", ds?.id, split, label], queryFn: () => api<{ total: number; items: Sample[] }>(`/api/datasets/${ds!.id}/samples${qs({ split, label, limit: 48 })}`), enabled: !!ds });
  const validate = useMutation({ mutationFn: () => api(`/api/datasets/${ds!.id}/validate`, { method: "POST" }), onSuccess: () => { qc.invalidateQueries({ queryKey: ["dataset", ds?.id] }); qc.invalidateQueries({ queryKey: ["datasets"] }); } });
  const d = detail.data;
  const qr = d?.quality_report as Record<string, any> | null | undefined;
  return (
    <Modal open={!!ds} onClose={onClose} title={ds ? `Dataset: ${ds.name}` : ""} wide>
      {!d ? <Spinner /> : (
        <div className="space-y-4">
          {d.synthetic && <Note tone="warn">Synthetic, procedurally generated images for demonstration. Not real production data.</Note>}
          <div className="grid gap-4 md:grid-cols-2">
            <KV items={[["Product", d.product_key], ["Kind / version", `${d.kind} · ${d.version}`], ["Samples", int(d.sample_count)], ["License", d.license_url ? <a className="underline" href={d.license_url} target="_blank" rel="noreferrer">{d.license}</a> : d.license ?? "—"],
              ["Source", d.source_url ? <a className="underline" href={d.source_url} target="_blank" rel="noreferrer">{d.source}</a> : d.source ?? "—"], ["Defect types annotated", d.defect_types_annotated ? "yes (from source labels)" : "no — binary normal/anomaly only"], ["Masks", d.masks_available ? "available" : "none"],
              ["Manifest SHA-256", <span className="font-mono text-[11px] break-all">{d.manifest_sha256}</span>]]} />
            <div className="space-y-2 text-sm">
              {d.citation && <div><p className="label">Citation</p><p className="text-xs">{d.citation}</p></div>}
              {d.split_policy && <div><p className="label">Split policy</p><p className="text-xs">{d.split_policy}</p></div>}
              {d.preprocessing && <div><p className="label">Preprocessing</p><p className="text-xs">{d.preprocessing}</p></div>}
            </div>
          </div>
          {qr ? (
            <div className="grid gap-4 md:grid-cols-3">
              <div className="card p-3 text-sm"><p className="label">Validation</p>{qr.valid ? <Badge tone="green"><ShieldCheck className="size-3" /> valid</Badge> : <Badge tone="red">invalid</Badge>}
                <p className="mt-2 text-xs">Readable {qr.num_readable}/{qr.num_samples} · errors {qr.errors?.length ?? 0} · mask issues {qr.mask_issues?.length ?? 0}</p>
                <p className="text-xs">Formats: {Object.entries(qr.formats ?? {}).map(([k, v]) => `${k} ${v}`).join(", ")}</p><p className="text-xs">Sizes: {Object.entries(qr.dimensions ?? {}).slice(0, 4).map(([k, v]) => `${k} (${v})`).join(", ")}</p></div>
              <div className="card p-3 text-sm"><p className="label">Balance & splits</p><p className="text-xs">normal : anomaly = {qr.normal_to_anomaly_ratio ?? "—"} : 1</p>
                {Object.entries(qr.splits ?? {}).map(([s, c]: [string, any]) => <p key={s} className="text-xs">{s}: {c.normal} normal / {c.anomaly} anomaly</p>)}
                <p className="mt-1 text-xs">{Object.entries(qr.defect_types ?? {}).map(([k, v]) => `${k} ${v}`).join(" · ")}</p></div>
              <div className="card p-3 text-sm"><p className="label">Duplicates & leakage</p>
                <p className="text-xs">Exact duplicate groups: {qr.exact_duplicate_groups?.length ?? 0}</p><p className="text-xs">Near-duplicate pairs (dHash): {qr.near_duplicate_pairs ?? 0}</p>
                <p className="text-xs">{qr.leakage?.leakage_free ? <Badge tone="green">no cross-split leakage</Badge> : <Badge tone="red">leakage detected</Badge>}</p></div>
            </div>
          ) : <Note>No quality report yet. Run validation.</Note>}
          {(qr?.warnings ?? d.warnings ?? []).length > 0 && <Note tone="warn"><ul className="list-disc pl-4">{(qr?.warnings ?? d.warnings).map((w: string, i: number) => <li key={i}>{w}</li>)}</ul></Note>}
          {can("admin") && <Button onClick={() => validate.mutate()} loading={validate.isPending}><ShieldCheck className="size-4" /> Re-run validation</Button>}
          <ErrorBox error={validate.error} />
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h3 className="text-sm font-semibold">Samples ({samples.data?.total ?? "…"})</h3>
              <select className="input w-32 py-1 text-xs" value={split} onChange={(e) => setSplit(e.target.value)} aria-label="Split"><option value="">all splits</option><option>train</option><option>val</option><option>test</option></select>
              <select className="input w-32 py-1 text-xs" value={label} onChange={(e) => setLabel(e.target.value)} aria-label="Label"><option value="">all labels</option><option>normal</option><option>anomaly</option></select>
              <label className="flex items-center gap-1 text-xs"><input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> show masks</label>
            </div>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
              {(samples.data?.items ?? []).map((s) => (
                <figure key={s.index} className="overflow-hidden rounded border border-slate-200 text-[10px] dark:border-slate-800">
                  <img src={`/api/datasets/${d.id}/samples/${s.index}/${showMask && s.has_mask ? "mask" : "image"}`} alt={s.path} loading="lazy" className="aspect-square w-full bg-slate-100 object-cover dark:bg-slate-800" />
                  <figcaption className="p-1"><Badge tone={s.label === "anomaly" ? "red" : "green"}>{s.label}</Badge> <span className="muted">{s.split}</span>{s.defect_type && <p className="truncate">{s.defect_type}</p>}</figcaption>
                </figure>
              ))}
            </div>
          </div>
        </div>
      )}
    </Modal>
  );
}

export default function Datasets() {
  const { can } = useAuth();
  const isAdmin = can("admin");
  const toast = useToast();
  const qc = useQueryClient();
  const [tab, setTab] = useState<"datasets" | "feedback">("datasets");
  const [open, setOpen] = useState<DatasetRec | null>(null);
  const [importOpen, setImportOpen] = useState(false);
  const [fbStatus, setFbStatus] = useState("pending");
  const [base, setBase] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const ds = useDatasets();
  const fb = useQuery({ queryKey: ["feedback", fbStatus], queryFn: () => api<{ items: Feedback[]; counts: Record<string, number> }>(`/api/feedback${qs({ status: fbStatus })}`) });
  const sync = useMutation({ mutationFn: () => api<{ count: number }>("/api/datasets/sync", { method: "POST" }), onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["datasets"] }); toast("ok", `${r.count} dataset(s) registered`); } });
  const decide = useMutation({ mutationFn: ({ id, approve }: { id: number; approve: boolean }) => api(`/api/feedback/${id}/decision`, { body: { approve } }), onSuccess: () => qc.invalidateQueries({ queryKey: ["feedback"] }), onError: (e: Error) => toast("error", e.message) });
  const build = useMutation({ mutationFn: () => api<Job>("/api/datasets/feedback-version", { body: { base_dataset_id: Number(base), seed: 42 } }), onSuccess: (j) => setJob(j.id) });
  return (
    <>
      <PageHeader title="Datasets & feedback" description="Registered datasets with licence, provenance, validation and leakage checks — and the human-approved feedback loop for controlled model improvement."
        actions={isAdmin && <><Button onClick={() => sync.mutate()} loading={sync.isPending}><RefreshCw className="size-4" /> Rescan</Button><Button variant="primary" onClick={() => setImportOpen(true)}><Upload className="size-4" /> Import dataset</Button></>} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "datasets", label: "Datasets" }, { id: "feedback", label: <>Feedback samples {fb.data?.counts.pending ? <Badge tone="amber">{fb.data.counts.pending}</Badge> : null}</> }]} />
      {tab === "datasets" && (
        <>
          <ErrorBox error={ds.error} />
          {ds.isLoading ? <Spinner /> : !ds.data?.length ? <div className="card"><Empty title="No datasets registered" icon={<Database className="size-8" />}>Run <code className="kbd">python -m app.cli bootstrap-demo</code> or import a dataset.</Empty></div> : (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {ds.data.map((d) => (
                <button key={d.id} onClick={() => setOpen(d)} className="card p-4 text-left hover:border-brand-500">
                  <div className="flex items-start justify-between gap-2"><div><p className="font-semibold">{d.name}</p><p className="muted text-xs">{d.product_key} · {d.version}</p></div>
                    <div className="flex flex-wrap justify-end gap-1">{d.synthetic && <Badge tone="violet">synthetic</Badge>}<Badge tone="slate">{d.kind}</Badge>{d.valid === false && <Badge tone="red">invalid</Badge>}</div></div>
                  <p className="mt-2 text-sm">{int(d.sample_count)} samples · {d.labels ? `${d.labels.normal ?? 0} normal / ${d.labels.anomaly ?? 0} anomaly` : "not validated"}</p>
                  <p className="muted mt-1 text-xs">License: {d.license ?? "unknown"}{d.masks_available ? " · masks" : ""}{d.defect_types_annotated ? " · defect types" : ""}</p>
                  {d.splits && <p className="muted mt-1 text-xs">{Object.entries(d.splits).map(([s, c]) => `${s} ${(c.normal ?? 0) + (c.anomaly ?? 0)}`).join(" · ")}</p>}
                  {d.warnings.length > 0 && <p className="mt-1 text-xs text-amber-600">{d.warnings.length} warning(s)</p>}
                </button>
              ))}
            </div>
          )}
        </>
      )}
      {tab === "feedback" && (
        <div className="space-y-4">
          <Note>Flow: inspector review → proposed feedback sample → <b>admin approval</b> → new versioned dataset (base dataset + approved samples, original test split untouched) → train a candidate → evaluate → approve → activate. Nothing is retrained or replaced automatically.</Note>
          {isAdmin && (
            <Card title="Build a new dataset version from approved feedback">
              <div className="flex flex-wrap items-end gap-3">
                <Field label="Base dataset">{(id) => <select id={id} className="input w-64" value={base} onChange={(e) => setBase(e.target.value)}><option value="">Choose…</option>{(ds.data ?? []).map((d) => <option key={d.id} value={d.id}>{d.name} ({d.product_key})</option>)}</select>}</Field>
                <Button variant="primary" disabled={!base} loading={build.isPending} onClick={() => build.mutate()}><GitBranchPlus className="size-4" /> Build version</Button>
              </div>
              <ErrorBox error={build.error} />
              {job && <div className="mt-3"><JobStatus jobId={job} /></div>}
            </Card>
          )}
          <Card bodyClass="p-0" title="Feedback samples" actions={<select className="input w-36 py-1 text-xs" value={fbStatus} onChange={(e) => setFbStatus(e.target.value)} aria-label="Status"><option value="pending">pending ({fb.data?.counts.pending ?? 0})</option><option value="approved">approved</option><option value="rejected">rejected</option><option value="used">used</option><option value="">all</option></select>}>
            {!fb.data?.items.length ? <p className="muted p-4 text-sm">No feedback samples with this status.</p> : (
              <table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Image</th><th className="th">Product</th><th className="th">Human label</th><th className="th">Model said</th><th className="th">Status</th><th className="th">Created</th><th className="th"></th></tr></thead>
                <tbody>{fb.data.items.map((f) => (
                  <tr key={f.id} className="border-b border-slate-100 dark:border-slate-800/60">
                    <td className="td">{f.thumbnail ? <Link to={`/inspections/${f.inspection_id}`}><img src={f.thumbnail} alt="" className="h-10 w-14 rounded object-cover" /></Link> : "—"}</td>
                    <td className="td">{f.product_key}</td>
                    <td className="td"><Badge tone={f.label === "defective" ? "red" : "green"}>{f.label}</Badge> {f.defect_type}</td>
                    <td className="td text-xs">{f.model_decision}</td>
                    <td className="td"><StatusBadge status={f.status} />{f.dataset_id && <span className="muted ml-1 text-xs">→ dataset #{f.dataset_id}</span>}</td>
                    <td className="td muted text-xs">{dt(f.created_at)}</td>
                    <td className="td">{isAdmin && f.status === "pending" && <div className="flex gap-1">
                      <Button size="sm" variant="success" onClick={() => decide.mutate({ id: f.id, approve: true })}><CheckCircle2 className="size-3.5" /> Approve</Button>
                      <Button size="sm" variant="danger" onClick={() => decide.mutate({ id: f.id, approve: false })}><XCircle className="size-3.5" /> Reject</Button></div>}</td>
                  </tr>
                ))}</tbody></table>
            )}
          </Card>
        </div>
      )}
      <DatasetDetail ds={open} onClose={() => setOpen(null)} />
      <ImportModal open={importOpen} onClose={() => setImportOpen(false)} />
    </>
  );
}

function ImportModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [product, setProduct] = useState("");
  const [license, setLicense] = useState("");
  const [job, setJob] = useState<string | null>(null);
  const qc = useQueryClient();
  const go = useMutation({
    mutationFn: () => { const fd = new FormData(); fd.append("file", file!); fd.append("name", name); fd.append("product_key", product || name); fd.append("license", license || "user-provided"); return api<Job>("/api/datasets/import", { form: fd }); },
    onSuccess: (j) => { setJob(j.id); setTimeout(() => qc.invalidateQueries({ queryKey: ["datasets"] }), 5000); },
  });
  const slug = /^[a-z0-9][a-z0-9_-]{1,63}$/;
  return (
    <Modal open={open} onClose={onClose} title="Import a custom dataset (ZIP)">
      <div className="space-y-3 text-sm">
        <Note>Accepted layouts: <code>normal/*.png</code> + <code>anomaly/&lt;defect_type&gt;/*.png</code> (optional <code>masks/</code> mirroring anomaly), or MVTec-style <code>train/good</code>, <code>test/&lt;type&gt;</code>, <code>ground_truth/&lt;type&gt;</code>. Images are validated and re-split reproducibly (train = normal only). Only import data you are licensed to use.</Note>
        <Field label="Dataset name (lowercase, a-z 0-9 _ -)">{(id) => <input id={id} className="input" value={name} onChange={(e) => setName(e.target.value)} />}</Field>
        <Field label="Product key (defaults to the dataset name)">{(id) => <input id={id} className="input" value={product} onChange={(e) => setProduct(e.target.value)} />}</Field>
        <Field label="License / usage terms">{(id) => <input id={id} className="input" value={license} onChange={(e) => setLicense(e.target.value)} placeholder="e.g. internal, CC-BY-4.0" />}</Field>
        <input type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] ?? null)} aria-label="ZIP file" />
        <ErrorBox error={go.error} />
        {job ? <JobStatus jobId={job} /> : <Button variant="primary" className="w-full" disabled={!file || !slug.test(name) || (!!product && !slug.test(product))} loading={go.isPending} onClick={() => go.mutate()}><Upload className="size-4" /> Upload & import</Button>}
      </div>
    </Modal>
  );
}
