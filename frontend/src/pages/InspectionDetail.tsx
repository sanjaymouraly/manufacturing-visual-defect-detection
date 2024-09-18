import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import InspectionResult from "../components/InspectionResult";
import { Badge, Card, ErrorBox, PageHeader, SeverityBadge, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { dt } from "../lib/format";
import type { Inspection } from "../lib/types";

export interface AuditRow { id: number; created_at: string; actor: string | null; action: string; details: Record<string, unknown> | null; hash: string }

export function AuditTrail({ rows }: { rows: AuditRow[] }) {
  if (!rows.length) return <p className="muted text-sm">No audit events.</p>;
  return (
    <ol className="space-y-2">
      {rows.map((e) => (
        <li key={e.id} className="flex gap-3 text-sm">
          <span className="mt-1.5 size-2 shrink-0 rounded-full bg-brand-600" />
          <div className="min-w-0">
            <p><span className="font-medium">{e.action}</span> <span className="muted">by {e.actor ?? "system"} · {dt(e.created_at)}</span></p>
            {e.details && Object.keys(e.details).length > 0 && <p className="muted truncate font-mono text-[11px]" title={JSON.stringify(e.details)}>{JSON.stringify(e.details)}</p>}
          </div>
        </li>
      ))}
    </ol>
  );
}

export default function InspectionDetail() {
  const { id } = useParams();
  const q = useQuery({ queryKey: ["inspection", id], queryFn: () => api<Inspection>(`/api/inspections/${id}`) });
  const audit = useQuery({ queryKey: ["inspection-audit", id], queryFn: () => api<AuditRow[]>(`/api/inspections/${id}/audit`) });
  return (
    <>
      <Link to="/history" className="muted mb-2 inline-flex items-center gap-1 text-sm hover:underline"><ArrowLeft className="size-4" /> History</Link>
      <PageHeader title={q.data ? q.data.filename : "Inspection"} description={q.data ? `Inspection ${q.data.id}` : undefined} />
      <ErrorBox error={q.error} />
      {q.isLoading && <Spinner />}
      {q.data && (
        <div className="space-y-4">
          <InspectionResult insp={q.data} />
          <div className="grid gap-4 lg:grid-cols-2">
            <Card title={`Human reviews (${q.data.reviews?.length ?? 0})`} subtitle="Reviews are append-only; the latest review determines the final label.">
              {q.data.reviews?.length ? (
                <ul className="space-y-3">
                  {[...q.data.reviews].reverse().map((r) => (
                    <li key={r.id} className="rounded-lg border border-slate-200 p-3 text-sm dark:border-slate-800">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={r.label === "defective" ? "red" : "green"}>{r.label}</Badge>
                        {r.defect_type && <Badge>{r.defect_type}</Badge>}
                        <SeverityBadge severity={r.severity} />
                        {r.agrees_with_model !== null && <Badge tone={r.agrees_with_model ? "teal" : "amber"}>{r.agrees_with_model ? "agrees with model" : "overrides model"}</Badge>}
                        <span className="muted ml-auto text-xs">{r.reviewer} · {dt(r.created_at)}</span>
                      </div>
                      {r.comment && <p className="mt-2">{r.comment}</p>}
                      {r.regions.length > 0 && <p className="muted mt-1 text-xs">{r.regions.length} marked region(s)</p>}
                    </li>
                  ))}
                </ul>
              ) : <p className="muted text-sm">Not reviewed yet.</p>}
            </Card>
            <Card title="Audit trail" subtitle="Hash-chained, append-only event log">
              {audit.data ? <AuditTrail rows={audit.data} /> : <Spinner />}
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
