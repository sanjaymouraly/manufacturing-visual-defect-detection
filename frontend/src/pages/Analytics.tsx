import { useMutation, useQuery } from "@tanstack/react-query";
import { Bot, FileDown, FileText, Send } from "lucide-react";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { JobStatus } from "../components/inspection";
import ProductSelect from "../components/ProductSelect";
import { Button, Card, Empty, ErrorBox, Field, Note, PageHeader, Spinner, Stat, StatusBadge } from "../components/ui";
import { api, download, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, int, ms, pct } from "../lib/format";
import { useToast } from "../lib/toast";
import type { AiText, Job } from "../lib/types";
import { chartColors } from "./Dashboard";

interface Full {
  window_days: number; n_inspections: number; n_reviewed: number; model_vs_human: Record<string, number>; false_alarm_rate_among_rejects: number | null;
  confirmed_defect_types: Record<string, number>; review_workload: { pending: number; completed: number; per_reviewer: Record<string, number>; median_turnaround_min: number | null };
  throughput: { per_hour: { hour: string; count: number }[]; latency_ms_mean: number | null; latency_ms_p95: number | null };
  batches: { batch_code: string; total: number; PASS: number; FAIL: number; REVIEW: number; fail_rate: number }[];
  by_model: { model: string; total: number; PASS?: number; FAIL?: number; REVIEW?: number }[];
}

function AiBox({ text }: { text: AiText }) {
  return (
    <div className="mt-3 rounded-lg border border-slate-200 p-3 dark:border-slate-800">
      <p className="muted mb-1 text-[11px]">{text.model_generated ? `AI-generated · ${text.generated_by}` : text.generated_by}</p>
      {text.model_generated && <Note tone="warn">Model-generated text based only on aggregated records (no images sent). Verify against the charts.</Note>}
      <p className="mt-2 text-sm whitespace-pre-wrap">{text.text}</p>
    </div>
  );
}

export default function Analytics() {
  const [product, setProduct] = useState("");
  const [days, setDays] = useState(30);
  const [question, setQuestion] = useState("");
  const [reportJob, setReportJob] = useState<string | null>(null);
  const { can } = useAuth();
  const toast = useToast();
  const q = useQuery({ queryKey: ["analytics", product, days], queryFn: () => api<Full>(`/api/analytics${qs({ product_key: product, days })}`) });
  const reports = useQuery({ queryKey: ["jobs", "report"], queryFn: () => api<Job[]>("/api/jobs?type=report&limit=10"), refetchInterval: reportJob ? 3000 : false });
  const summary = useMutation({ mutationFn: () => api<AiText>(`/api/ai/period-summary${qs({ product_key: product, days })}`, { method: "POST" }) });
  const ask = useMutation({ mutationFn: () => api<AiText>("/api/ai/ask", { body: { question, product_key: product || null, days } }) });
  const gen = useMutation({
    mutationFn: () => api<Job>("/api/reports", { body: { days, product_key: product || null } }),
    onSuccess: (j) => { setReportJob(j.id); toast("ok", "Report generation started"); },
  });
  const d = q.data;
  const mvh = d?.model_vs_human ?? {};
  const confusion = [
    ["Model FAIL", mvh.true_reject ?? 0, mvh.false_alarm ?? 0],
    ["Model PASS", mvh.missed_defect ?? 0, mvh.true_accept ?? 0],
    ["Model REVIEW", mvh.review_defective ?? 0, mvh.review_normal ?? 0],
  ] as const;
  return (
    <>
      <PageHeader title="Analytics & reports" description="Production analytics from recorded inspections and human reviews. Model-vs-human figures only cover reviewed items."
        actions={<>
          <ProductSelect value={product} onChange={setProduct} allowAll className="input w-56" />
          <select className="input w-36" value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Time window">{[1, 7, 30, 90, 365].map((n) => <option key={n} value={n}>Last {n} days</option>)}</select>
        </>} />
      <ErrorBox error={q.error} />
      {q.isLoading && <Spinner />}
      {d && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Inspections" value={int(d.n_inspections)} />
            <Stat label="Human-reviewed" value={int(d.n_reviewed)} />
            <Stat label="False alarms among rejects" value={pct(d.false_alarm_rate_among_rejects)} hint="reviewed FAILs judged normal" />
            <Stat label="Missed defects (reviewed)" value={int(mvh.missed_defect ?? 0)} tone={(mvh.missed_defect ?? 0) > 0 ? "red" : undefined} hint="PASS later judged defective" />
            <Stat label="Mean latency" value={ms(d.throughput.latency_ms_mean)} hint={`p95 ${ms(d.throughput.latency_ms_p95)}`} />
            <Stat label="Review backlog" value={int(d.review_workload.pending)} hint={`median turnaround ${d.review_workload.median_turnaround_min ?? "—"} min`} />
          </div>
          <div className="grid gap-5 lg:grid-cols-2">
            <Card title="Model decision vs. human label" subtitle="Counts of reviewed inspections">
              {d.n_reviewed === 0 ? <Empty title="No reviewed inspections yet" /> : (
                <table className="w-full text-sm">
                  <thead><tr><th className="th"></th><th className="th text-center">Human: defective</th><th className="th text-center">Human: normal</th></tr></thead>
                  <tbody>{confusion.map(([k, a, b]) => (
                    <tr key={k} className="border-t border-slate-100 dark:border-slate-800"><td className="td font-medium">{k}</td>
                      <td className={`td text-center tabular-nums ${k === "Model PASS" && a > 0 ? "font-bold text-red-600" : ""}`}>{a}</td>
                      <td className={`td text-center tabular-nums ${k === "Model FAIL" && b > 0 ? "text-amber-600" : ""}`}>{b}</td></tr>
                  ))}</tbody>
                </table>
              )}
            </Card>
            <Card title="Confirmed defect types">
              {Object.keys(d.confirmed_defect_types).length === 0 ? <Empty title="No confirmed defects yet" /> : (
                <div className="h-56"><ResponsiveContainer><BarChart data={Object.entries(d.confirmed_defect_types).map(([k, v]) => ({ type: k, count: v }))} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis type="number" allowDecimals={false} fontSize={11} /><YAxis dataKey="type" type="category" width={120} fontSize={11} /><Tooltip /><Bar dataKey="count" fill="#ef4444" />
                </BarChart></ResponsiveContainer></div>
              )}
            </Card>
            <Card title="Throughput (inspections per hour)">
              {d.throughput.per_hour.length === 0 ? <Empty title="No data" /> : (
                <div className="h-56"><ResponsiveContainer><LineChart data={d.throughput.per_hour}><CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis dataKey="hour" fontSize={10} /><YAxis allowDecimals={false} fontSize={11} /><Tooltip /><Line dataKey="count" stroke="#0d9488" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div>
              )}
            </Card>
            <Card title="Decisions by model">
              {d.by_model.length === 0 ? <Empty title="No data" /> : (
                <div className="h-56"><ResponsiveContainer><BarChart data={d.by_model}><CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} /><XAxis dataKey="model" fontSize={10} /><YAxis allowDecimals={false} fontSize={11} /><Tooltip />
                  {(["PASS", "REVIEW", "FAIL"] as const).map((k) => <Bar key={k} dataKey={k} stackId="a" fill={chartColors[k]} />)}</BarChart></ResponsiveContainer></div>
              )}
            </Card>
          </div>
          <Card title="Batches" bodyClass="p-0">
            {d.batches.length === 0 ? <p className="muted p-4 text-sm">No batch codes recorded in this window.</p> : (
              <div className="max-h-80 overflow-auto"><table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Batch</th><th className="th text-right">Total</th><th className="th text-right">Pass</th><th className="th text-right">Fail</th><th className="th text-right">Review</th><th className="th text-right">Fail rate</th></tr></thead>
                <tbody>{[...d.batches].sort((a, b) => b.fail_rate - a.fail_rate).map((b) => (
                  <tr key={b.batch_code} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td font-medium">{b.batch_code}</td><td className="td text-right">{b.total}</td><td className="td text-right">{b.PASS}</td><td className="td text-right">{b.FAIL}</td><td className="td text-right">{b.REVIEW}</td><td className="td text-right font-semibold">{pct(b.fail_rate)}</td></tr>
                ))}</tbody></table></div>
            )}
          </Card>
          <div className="grid gap-5 lg:grid-cols-2">
            <Card title="Reports" subtitle="PDF summary of the selected window, generated in the background">
              {can("admin", "inspector") && <Button variant="primary" onClick={() => gen.mutate()} loading={gen.isPending}><FileText className="size-4" /> Generate PDF report</Button>}
              <ErrorBox error={gen.error} />
              {reportJob && <div className="mt-3"><JobStatus jobId={reportJob} /></div>}
              <ul className="mt-3 divide-y divide-slate-100 text-sm dark:divide-slate-800">
                {(reports.data ?? []).map((j) => (
                  <li key={j.id} className="flex items-center gap-2 py-2">
                    <StatusBadge status={j.status} /><span className="flex-1 truncate">{j.params.product_key || "all products"} · {j.params.days} days</span><span className="muted text-xs">{ago(j.created_at)}</span>
                    {j.result?.file && <Button size="sm" onClick={() => download(`/api/reports/${j.result!.file}`, j.result!.file).catch((e) => toast("error", e.message))}><FileDown className="size-3.5" /> PDF</Button>}
                  </li>
                ))}
              </ul>
            </Card>
            <Card title="Ask about this period" subtitle="Uses the configured AI provider, or a deterministic summary in local mode">
              <div className="flex flex-wrap gap-2">
                <Button onClick={() => summary.mutate()} loading={summary.isPending}><Bot className="size-4" /> Summarize period</Button>
              </div>
              <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); if (question.trim()) ask.mutate(); }}>
                <Field label="Question" className="flex-1">{(id) => <input id={id} className="input" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="e.g. Which batch had the most rejects?" maxLength={500} />}</Field>
                <Button type="submit" className="self-end" loading={ask.isPending} disabled={!question.trim()}><Send className="size-4" /> Ask</Button>
              </form>
              <ErrorBox error={summary.error || ask.error} title="AI request failed" />
              {summary.data && <AiBox text={summary.data} />}
              {ask.data && <AiBox text={ask.data} />}
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
