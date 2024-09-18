import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { FileDown, Search, X } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Badge, Button, Card, DecisionBadge, Empty, ErrorBox, PageHeader, Pagination, SeverityBadge, Spinner } from "../components/ui";
import { api, download, qs } from "../lib/api";
import { dt, num } from "../lib/format";
import { useToast } from "../lib/toast";
import type { Inspection, Paged } from "../lib/types";

interface Facets { batch_codes: string[]; camera_ids: string[]; products: string[]; reviewers: string[]; models: { id: number; name: string }[] }
const KEYS = ["q", "product_key", "decision", "review_status", "final_label", "batch_code", "camera_id", "model_id", "source", "reviewer", "date_from", "date_to", "job_id"] as const;

export default function HistoryPage() {
  const [sp, setSp] = useSearchParams();
  const [page, setPage] = useState(1);
  const [text, setText] = useState(sp.get("q") ?? "");
  const nav = useNavigate();
  const toast = useToast();
  const filters = Object.fromEntries(KEYS.map((k) => [k, sp.get(k) ?? ""])) as Record<(typeof KEYS)[number], string>;
  const set = (k: string, v: string) => { const n = new URLSearchParams(sp); if (v) n.set(k, v); else n.delete(k); setSp(n); setPage(1); };
  const facets = useQuery({ queryKey: ["facets"], queryFn: () => api<Facets>("/api/inspections/facets") });
  const query = { ...filters, date_from: filters.date_from ? `${filters.date_from}T00:00:00` : "", date_to: filters.date_to ? `${filters.date_to}T23:59:59` : "" };
  const q = useQuery({ queryKey: ["history", query, page], queryFn: () => api<Paged<Inspection>>(`/api/inspections${qs({ ...query, page, page_size: 25 })}`), placeholderData: keepPreviousData });
  const active = KEYS.filter((k) => filters[k]).length;
  const sel = (k: (typeof KEYS)[number], label: string, opts: (string | { v: string; l: string })[]) => (
    <select className="input py-1.5 text-xs" value={filters[k]} onChange={(e) => set(k, e.target.value)} aria-label={label}>
      <option value="">{label}: all</option>
      {opts.map((o) => typeof o === "string" ? <option key={o} value={o}>{o}</option> : <option key={o.v} value={o.v}>{o.l}</option>)}
    </select>
  );
  return (
    <>
      <PageHeader title="Inspection history" description="Search and filter all recorded inspections. Exports respect the current filters (CSV cells are escaped against spreadsheet formula injection)."
        actions={<Button onClick={() => download(`/api/inspections/export.csv${qs(query)}`, "inspections.csv").catch((e) => toast("error", e.message))}><FileDown className="size-4" /> Export CSV</Button>} />
      <Card className="mb-4">
        <form className="mb-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); set("q", text); }}>
          <div className="relative flex-1"><Search className="absolute top-2.5 left-2.5 size-4 text-slate-400" /><input className="input pl-8" placeholder="Search file name, inspection ID, external reference, batch…" value={text} onChange={(e) => setText(e.target.value)} aria-label="Search" /></div>
          <Button type="submit">Search</Button>
        </form>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-6">
          {sel("product_key", "Product", facets.data?.products ?? [])}
          {sel("decision", "Decision", ["PASS", "FAIL", "REVIEW"])}
          {sel("review_status", "Review", ["pending", "reviewed", "none"])}
          {sel("final_label", "Final label", ["defective", "normal"])}
          {sel("batch_code", "Batch", facets.data?.batch_codes ?? [])}
          {sel("model_id", "Model", (facets.data?.models ?? []).map((m) => ({ v: String(m.id), l: m.name })))}
          {sel("source", "Source", ["upload", "batch", "camera", "api"])}
          {sel("camera_id", "Camera", facets.data?.camera_ids ?? [])}
          {sel("reviewer", "Reviewer", facets.data?.reviewers ?? [])}
          <input type="date" className="input py-1.5 text-xs" value={filters.date_from} onChange={(e) => set("date_from", e.target.value)} aria-label="From date" />
          <input type="date" className="input py-1.5 text-xs" value={filters.date_to} onChange={(e) => set("date_to", e.target.value)} aria-label="To date" />
          {active > 0 && <Button size="sm" variant="ghost" onClick={() => { setSp({}); setText(""); }}><X className="size-3.5" /> Clear {active} filter(s)</Button>}
        </div>
      </Card>
      <ErrorBox error={q.error} />
      <Card bodyClass="p-0">
        {q.isLoading ? <Spinner /> : !q.data?.items.length ? <Empty title="No inspections match" /> : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Time</th><th className="th">Image</th><th className="th">File / ref</th><th className="th">Product · model</th><th className="th">Decision</th><th className="th">Severity</th><th className="th">Review</th><th className="th text-right">Score</th><th className="th text-right">Norm.</th></tr></thead>
                <tbody>
                  {q.data.items.map((i) => (
                    <tr key={i.id} className="cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-800/40" onClick={() => nav(`/inspections/${i.id}`)}>
                      <td className="td muted text-xs whitespace-nowrap">{dt(i.created_at)}</td>
                      <td className="td">{i.images.display ? <img src={i.images.overlay ?? i.images.display} alt="" className="h-9 w-12 rounded object-cover" loading="lazy" /> : <div className="h-9 w-12 rounded bg-slate-200 dark:bg-slate-800" />}</td>
                      <td className="td max-w-56"><Link to={`/inspections/${i.id}`} className="block truncate hover:underline" onClick={(e) => e.stopPropagation()}>{i.filename}</Link><span className="muted text-xs">{[i.batch_code, i.external_ref, i.source].filter(Boolean).join(" · ")}</span></td>
                      <td className="td text-xs">{i.product_key}<p className="muted truncate">{i.model_name}</p></td>
                      <td className="td"><DecisionBadge decision={i.decision} />{i.status === "error" && <Badge tone="red">error</Badge>}</td>
                      <td className="td"><SeverityBadge severity={i.severity} /></td>
                      <td className="td">{i.review_status === "pending" ? <Badge tone="amber">pending</Badge> : i.final_label ? <Badge tone={i.final_label === "defective" ? "red" : "green"}>{i.final_label}</Badge> : <span className="muted text-xs">—</span>}</td>
                      <td className="td text-right tabular-nums">{num(i.score)}</td>
                      <td className="td text-right tabular-nums">{num(i.normalized_score, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} pageSize={25} total={q.data.total} onPage={setPage} />
          </>
        )}
      </Card>
    </>
  );
}
