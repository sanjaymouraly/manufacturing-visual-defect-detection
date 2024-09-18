import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileArchive, FileDown, Play, RotateCcw, UploadCloud, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { JobStatus, useJob } from "../components/inspection";
import ProductSelect from "../components/ProductSelect";
import { Badge, Button, Card, DecisionBadge, Empty, ErrorBox, Field, Note, PageHeader, SeverityBadge, Stat, StatusBadge } from "../components/ui";
import { api, download, qs } from "../lib/api";
import { ago, int, num, pct } from "../lib/format";
import { useToast } from "../lib/toast";
import type { Inspection, Job } from "../lib/types";

export default function Batch() {
  const [params, setParams] = useSearchParams();
  const jobId = params.get("job");
  const [product, setProduct] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [batch, setBatch] = useState("");
  const [station, setStation] = useState("");
  const [force, setForce] = useState(false);
  const [filter, setFilter] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const qc = useQueryClient();
  const toast = useToast();
  const job = useJob(jobId);
  const done = job.data && ["succeeded", "failed", "cancelled"].includes(job.data.status);
  const results = useQuery({
    queryKey: ["batch-results", jobId, done],
    queryFn: () => api<{ job: Job; items: Inspection[] }>(`/api/batches/${jobId}/results`),
    enabled: !!jobId,
    refetchInterval: done ? false : 2000,
  });
  const recent = useQuery({ queryKey: ["jobs", "batch"], queryFn: () => api<Job[]>("/api/jobs?type=batch_inspection&limit=10"), refetchInterval: 10000 });
  useEffect(() => { if (done) qc.invalidateQueries({ queryKey: ["jobs", "batch"] }); }, [done, qc]);

  const start = useMutation({
    mutationFn: () => {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("product_key", product);
      fd.append("batch_code", batch);
      fd.append("station", station);
      fd.append("force", String(force));
      return api<Job>("/api/batches", { form: fd });
    },
    onSuccess: (j) => { setParams({ job: j.id }); setFiles([]); toast("ok", "Batch queued"); },
  });
  const cancel = useMutation({ mutationFn: () => api(`/api/jobs/${jobId}/cancel`, { method: "POST" }), onSuccess: () => job.refetch() });
  const retry = useMutation({ mutationFn: () => api(`/api/jobs/${jobId}/retry`, { method: "POST" }), onSuccess: () => job.refetch() });

  const items = useMemo(() => (results.data?.items ?? []).filter((i) => !filter || i.decision === filter), [results.data, filter]);
  const r = job.data?.result as { total?: number; PASS?: number; FAIL?: number; REVIEW?: number; errors?: number; duplicates?: number; skipped?: string[] } | null | undefined;
  const totalSize = files.reduce((a, f) => a + f.size, 0);

  return (
    <>
      <PageHeader title="Batch inspection" description="Upload many images or ZIP archives. Processing runs in the background, survives page reloads, and can be cancelled or retried. ZIP archives are checked for path traversal, symlinks and decompression bombs." />
      <div className="grid gap-5 xl:grid-cols-[380px_1fr]">
        <div className="space-y-5">
          <Card title="New batch">
            <div className="space-y-3">
              <Field label="Product">{(id) => <ProductSelect id={id} value={product} onChange={setProduct} requireActive />}</Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Batch / lot code">{(id) => <input id={id} className="input" value={batch} onChange={(e) => setBatch(e.target.value)} />}</Field>
                <Field label="Station">{(id) => <input id={id} className="input" value={station} onChange={(e) => setStation(e.target.value)} />}</Field>
              </div>
              <button type="button" onClick={() => input.current?.click()} onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => { e.preventDefault(); const dropped = Array.from(e.dataTransfer.files); setFiles((f) => [...f, ...dropped]); }}
                className="flex w-full flex-col items-center gap-2 rounded-xl border-2 border-dashed border-slate-300 p-5 text-sm dark:border-slate-700">
                <UploadCloud className="size-7 text-slate-400" />Drop images / ZIP files or click to choose
              </button>
              <input ref={input} type="file" multiple accept="image/*,.zip,.tif,.tiff,.bmp" className="hidden" data-testid="batch-input"
                onChange={(e) => {
                  // Copy before resetting: the FileList is live and is emptied by `value = ""`, while the
                  // state updater below only runs later, during the next render.
                  const picked = Array.from(e.target.files ?? []);
                  e.target.value = "";
                  setFiles((f) => [...f, ...picked]);
                }} />
              {files.length > 0 && (
                <div className="max-h-40 overflow-y-auto rounded-lg border border-slate-200 text-xs dark:border-slate-800">
                  {files.map((f, i) => (
                    <div key={i} className="flex items-center gap-2 border-b border-slate-100 px-2 py-1 last:border-0 dark:border-slate-800">
                      {f.name.toLowerCase().endsWith(".zip") && <FileArchive className="size-3.5 text-amber-600" />}
                      <span className="flex-1 truncate">{f.name}</span><span className="muted">{(f.size / 1024).toFixed(0)} KB</span>
                      <button aria-label={`Remove ${f.name}`} onClick={() => setFiles((x) => x.filter((_, j) => j !== i))}><X className="size-3.5" /></button>
                    </div>
                  ))}
                </div>
              )}
              {files.length > 0 && <p className="muted text-xs">{files.length} file(s) · {(totalSize / 1024 / 1024).toFixed(1)} MB</p>}
              <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /> Re-inspect duplicates (otherwise identical images reuse earlier results)</label>
              <Button variant="primary" className="w-full" disabled={!files.length || !product} loading={start.isPending} onClick={() => start.mutate()}><Play className="size-4" /> Start batch</Button>
              <ErrorBox error={start.error} title="Batch not started" />
            </div>
          </Card>
          <Card title="Recent batches" bodyClass="p-0">
            {(recent.data ?? []).length === 0 ? <p className="muted p-4 text-sm">No batches yet.</p> : (
              <ul>{recent.data!.map((j) => (
                <li key={j.id}>
                  <button onClick={() => setParams({ job: j.id })} className={`flex w-full items-center gap-2 border-b border-slate-100 px-4 py-2 text-left text-sm hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/40 ${j.id === jobId ? "bg-brand-50 dark:bg-brand-800/20" : ""}`}>
                    <StatusBadge status={j.status} /><span className="flex-1 truncate">{j.params.batch_code || j.params.product_key} · {j.params.n_files ?? "?"} file(s)</span><span className="muted text-xs">{ago(j.created_at)}</span>
                  </button>
                </li>
              ))}</ul>
            )}
          </Card>
        </div>
        <div className="min-w-0 space-y-4">
          {!jobId && <div className="card"><Empty title="No batch selected" icon={<UploadCloud className="size-8" />}>Start a new batch or open a recent one.</Empty></div>}
          {jobId && (
            <>
              <JobStatus jobId={jobId} onCancel={() => cancel.mutate()} />
              {job.data?.status === "failed" || job.data?.status === "cancelled" ? <Button onClick={() => retry.mutate()} loading={retry.isPending}><RotateCcw className="size-4" /> Retry / resume</Button> : null}
              {r && (
                <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                  <Stat label="Images" value={int(r.total)} />
                  <Stat label="Pass" value={int(r.PASS)} tone="green" />
                  <Stat label="Fail" value={int(r.FAIL)} tone="red" hint={r.total ? `${pct((r.FAIL ?? 0) / r.total)} of batch` : undefined} />
                  <Stat label="Review" value={int(r.REVIEW)} tone="amber" />
                  <Stat label="Errors / duplicates" value={`${r.errors ?? 0} / ${r.duplicates ?? 0}`} hint="errors are routed to review" />
                </div>
              )}
              {!!(job.data?.params.skipped?.length || r?.skipped?.length) && (
                <Note tone="warn"><p className="font-medium">Skipped archive entries</p><ul className="list-disc pl-4">{[...(job.data?.params.skipped ?? []), ...(r?.skipped ?? [])].slice(0, 20).map((s: string, i: number) => <li key={i}>{s}</li>)}</ul></Note>
              )}
              <Card title={`Results (${results.data?.items.length ?? 0})`} bodyClass="p-0"
                actions={<>
                  <select className="input w-32 py-1 text-xs" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter by decision"><option value="">All decisions</option><option>PASS</option><option>FAIL</option><option>REVIEW</option></select>
                  <Button size="sm" onClick={() => download(`/api/inspections/export.csv${qs({ job_id: jobId })}`, `batch-${jobId.slice(0, 8)}.csv`).catch((e) => toast("error", e.message))}><FileDown className="size-3.5" /> CSV</Button>
                </>}>
                <div className="max-h-[60vh] overflow-auto">
                  <table className="w-full">
                    <thead className="sticky top-0 bg-white dark:bg-slate-900"><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Image</th><th className="th">File</th><th className="th">Decision</th><th className="th">Severity</th><th className="th text-right">Norm. score</th><th className="th">Notes</th></tr></thead>
                    <tbody>
                      {items.map((i) => (
                        <tr key={i.id} className="border-b border-slate-100 dark:border-slate-800/60">
                          <td className="td">{i.images.overlay || i.images.display ? <img src={(i.images.overlay ?? i.images.display)!} alt="" className="h-10 w-14 rounded object-cover" loading="lazy" /> : null}</td>
                          <td className="td max-w-64 truncate"><Link to={`/inspections/${i.id}`} className="hover:underline">{i.filename}</Link></td>
                          <td className="td"><DecisionBadge decision={i.decision} /></td>
                          <td className="td"><SeverityBadge severity={i.severity} /></td>
                          <td className="td text-right tabular-nums">{num(i.normalized_score, 2)}</td>
                          <td className="td text-xs">{i.status === "error" ? <Badge tone="red">{i.error?.slice(0, 60) ?? "error"}</Badge> : i.duplicate_of ? <Badge tone="blue">duplicate</Badge> : i.review_status === "pending" ? <Badge tone="amber">needs review</Badge> : null}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {items.length === 0 && <p className="muted p-4 text-sm">{done ? "No results match." : "Results appear here as images are processed…"}</p>}
                </div>
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
