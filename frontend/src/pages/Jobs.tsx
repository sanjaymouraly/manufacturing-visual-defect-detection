import { useMutation, useQuery } from "@tanstack/react-query";
import { ListTodo, RotateCcw, Square } from "lucide-react";
import { useState } from "react";
import { Button, Card, Empty, ErrorBox, Modal, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { dt, titleCase } from "../lib/format";
import { useToast } from "../lib/toast";
import type { Job } from "../lib/types";

const TYPES = ["batch_inspection", "train_model", "evaluate_models", "robustness", "domain_shift", "train_classifier", "export_onnx", "report", "retention", "import_dataset", "build_feedback_dataset"];

export default function Jobs() {
  const [type, setType] = useState("");
  const [status, setStatus] = useState("");
  const [open, setOpen] = useState<Job | null>(null);
  const { can } = useAuth();
  const toast = useToast();
  const q = useQuery({ queryKey: ["jobs", type, status], queryFn: () => api<Job[]>(`/api/jobs${qs({ type, status, limit: 100 })}`), refetchInterval: 3000 });
  const act = useMutation({ mutationFn: ({ id, a }: { id: string; a: "cancel" | "retry" }) => api(`/api/jobs/${id}/${a}`, { method: "POST" }), onSuccess: () => q.refetch(), onError: (e: Error) => toast("error", e.message) });
  const duration = (j: Job) => j.started_at ? `${(((j.finished_at ? new Date(j.finished_at + (j.finished_at.includes("+") ? "" : "Z")) : new Date()).getTime() - new Date(j.started_at + (j.started_at.includes("+") ? "" : "Z")).getTime()) / 1000).toFixed(1)} s` : "—";
  return (
    <>
      <PageHeader title="Background jobs" description="Persistent job queue: jobs survive restarts (interrupted jobs are re-queued on startup), report progress, and can be cancelled or retried."
        actions={<>
          <select className="input w-52" value={type} onChange={(e) => setType(e.target.value)} aria-label="Type"><option value="">All types</option>{TYPES.map((t) => <option key={t} value={t}>{titleCase(t)}</option>)}</select>
          <select className="input w-40" value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status"><option value="">All statuses</option>{["queued", "running", "succeeded", "failed", "cancelled"].map((s) => <option key={s}>{s}</option>)}</select>
        </>} />
      <ErrorBox error={q.error} />
      <Card bodyClass="p-0">
        {q.isLoading ? <Spinner /> : !q.data?.length ? <Empty title="No jobs" icon={<ListTodo className="size-8" />} /> : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Created</th><th className="th">Type</th><th className="th">Status</th><th className="th w-64">Progress</th><th className="th">Attempts</th><th className="th">Duration</th><th className="th"></th></tr></thead>
              <tbody>{q.data.map((j) => (
                <tr key={j.id} className="border-b border-slate-100 dark:border-slate-800/60">
                  <td className="td muted text-xs whitespace-nowrap">{dt(j.created_at)}</td>
                  <td className="td"><button className="font-medium hover:underline" onClick={() => setOpen(j)}>{titleCase(j.type)}</button><p className="muted font-mono text-[10px]">{j.id.slice(0, 12)}</p></td>
                  <td className="td"><StatusBadge status={j.status} />{j.cancel_requested && j.status === "running" && <span className="muted ml-1 text-xs">cancelling</span>}</td>
                  <td className="td"><ProgressBar value={j.progress} label={`${j.type} progress`} /><p className="muted mt-1 truncate text-xs" title={j.error ?? j.message}>{j.error ?? j.message}</p></td>
                  <td className="td text-sm">{j.attempts}/{j.max_attempts}</td>
                  <td className="td text-sm tabular-nums">{duration(j)}</td>
                  <td className="td">{can("admin", "inspector") && (
                    <div className="flex gap-1">
                      {(j.status === "queued" || j.status === "running") && <Button size="sm" variant="ghost" onClick={() => act.mutate({ id: j.id, a: "cancel" })} disabled={j.cancel_requested}><Square className="size-3.5" /> Cancel</Button>}
                      {(j.status === "failed" || j.status === "cancelled") && <Button size="sm" variant="ghost" onClick={() => act.mutate({ id: j.id, a: "retry" })}><RotateCcw className="size-3.5" /> Retry</Button>}
                    </div>
                  )}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </Card>
      <Modal open={!!open} onClose={() => setOpen(null)} title={open ? `${titleCase(open.type)} · ${open.id}` : ""} wide>
        {open && (
          <div className="grid gap-3 md:grid-cols-2">
            <div><p className="label">Parameters</p><pre className="max-h-96 overflow-auto rounded bg-slate-100 p-2 text-[11px] dark:bg-slate-800">{JSON.stringify(open.params, null, 2)}</pre></div>
            <div><p className="label">Result</p><pre className="max-h-96 overflow-auto rounded bg-slate-100 p-2 text-[11px] dark:bg-slate-800">{JSON.stringify(open.result ?? open.error, null, 2)}</pre></div>
          </div>
        )}
      </Modal>
    </>
  );
}
