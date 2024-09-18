import { useQuery } from "@tanstack/react-query";
import { ClipboardCheck, FlaskConical, ScanSearch, UploadCloud } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import ProductSelect from "../components/ProductSelect";
import { Badge, Card, DecisionBadge, Empty, ErrorBox, Note, PageHeader, SeverityBadge, Spinner, Stat } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, int, num, pct } from "../lib/format";

interface Overview {
  window_days: number; total: number; pass: number; fail: number; review: number; errors: number; pending_reviews: number;
  anomaly_rate: number | null; review_rate: number | null; trend: { date: string; PASS: number; FAIL: number; REVIEW: number }[];
  recent: { id: string; created_at: string; product_key: string; decision: string; score: number | null; normalized_score: number | null; filename: string; severity: string }[];
  active_models: { product: string; product_name: string; synthetic: boolean; model: { id: number; name: string; algorithm: string; version: number; test_auroc: number | null } | null }[];
}

export const chartColors = { PASS: "#10b981", FAIL: "#ef4444", REVIEW: "#f59e0b" };

export default function Dashboard() {
  const [product, setProduct] = useState("");
  const [days, setDays] = useState(30);
  const { can } = useAuth();
  const nav = useNavigate();
  const q = useQuery({ queryKey: ["dashboard", product, days], queryFn: () => api<Overview>(`/api/dashboard${qs({ product_key: product, days })}`), refetchInterval: 20000 });
  const d = q.data;
  return (
    <>
      <PageHeader title="Dashboard" description="Inspection outcomes, review workload and active models. All figures are computed from recorded inspections."
        actions={<>
          <ProductSelect value={product} onChange={setProduct} allowAll className="input w-56" />
          <select className="input w-36" value={days} onChange={(e) => setDays(Number(e.target.value))} aria-label="Time window">
            {[1, 7, 30, 90, 365].map((n) => <option key={n} value={n}>Last {n} day{n > 1 ? "s" : ""}</option>)}
          </select>
        </>} />
      <ErrorBox error={q.error} />
      {q.isLoading && <Spinner />}
      {d && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Inspected" value={int(d.total)} hint={`last ${d.window_days} days`} />
            <Stat label="Pass" value={int(d.pass)} tone="green" />
            <Stat label="Fail" value={int(d.fail)} tone="red" hint={`reject rate ${pct(d.anomaly_rate)}`} />
            <Stat label="Review" value={int(d.review)} tone="amber" hint={`review rate ${pct(d.review_rate)}`} />
            <Stat label="Pending reviews" value={int(d.pending_reviews)} hint={<Link className="underline" to="/reviews">open queue</Link>} />
            <Stat label="Could not inspect" value={int(d.errors)} hint="routed to review (fail-safe)" />
          </div>
          {can("admin", "inspector") && (
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {[
                { to: "/inspect", icon: <ScanSearch className="size-5" />, label: "Inspect an image" },
                { to: "/batch", icon: <UploadCloud className="size-5" />, label: "Batch / ZIP upload" },
                { to: "/reviews", icon: <ClipboardCheck className="size-5" />, label: "Review queue" },
                { to: "/models", icon: <FlaskConical className="size-5" />, label: "Model lab" },
              ].map((a) => (
                <Link key={a.to} to={a.to} className="card flex items-center gap-3 p-3 text-sm font-medium hover:border-brand-500">
                  <span className="grid size-9 place-items-center rounded-lg bg-brand-50 text-brand-700 dark:bg-brand-800/30 dark:text-brand-500">{a.icon}</span>{a.label}
                </Link>
              ))}
            </div>
          )}
          <div className="grid gap-5 xl:grid-cols-3">
            <Card title="Decisions per day" className="xl:col-span-2">
              {d.trend.length === 0 ? <Empty title="No inspections in this window" /> : (
                <div className="h-64">
                  <ResponsiveContainer>
                    <BarChart data={d.trend}>
                      <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.2} />
                      <XAxis dataKey="date" fontSize={11} />
                      <YAxis allowDecimals={false} fontSize={11} />
                      <Tooltip contentStyle={{ fontSize: 12 }} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      {(["PASS", "REVIEW", "FAIL"] as const).map((k) => <Bar key={k} dataKey={k} stackId="a" fill={chartColors[k]} />)}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>
            <Card title="Active models" actions={<Link to="/models" className="text-xs underline">Model lab</Link>}>
              <ul className="space-y-2">
                {d.active_models.map((a) => (
                  <li key={a.product} className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 p-2 text-sm dark:border-slate-800">
                    <div className="min-w-0">
                      <p className="truncate font-medium">{a.product_name} {a.synthetic && <Badge tone="violet">synthetic demo</Badge>}</p>
                      <p className="muted truncate text-xs">{a.model ? `${a.model.name} · v${a.model.version}` : "No active model — inspections are not possible"}</p>
                    </div>
                    {a.model ? <Badge tone="teal" title="Image AUROC on held-out test split">AUROC {num(a.model.test_auroc)}</Badge> : <Badge tone="amber">inactive</Badge>}
                  </li>
                ))}
              </ul>
              <div className="mt-3"><Note>Synthetic demo products use generated images for demonstration only; their metrics do not reflect real production performance.</Note></div>
            </Card>
          </div>
          <Card title="Recent inspections" actions={<Link to="/history" className="text-xs underline">Full history</Link>} bodyClass="p-0">
            {d.recent.length === 0 ? <Empty title="No inspections yet" icon={<ScanSearch className="size-8" />}>Upload an image on the Inspect page to get started.</Empty> : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Time</th><th className="th">Product</th><th className="th">File</th><th className="th">Decision</th><th className="th">Severity</th><th className="th text-right">Score / thr.</th></tr></thead>
                  <tbody>
                    {d.recent.map((r) => (
                      <tr key={r.id} className="cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-800/40" onClick={() => nav(`/inspections/${r.id}`)}>
                        <td className="td muted whitespace-nowrap">{ago(r.created_at)}</td>
                        <td className="td">{r.product_key}</td>
                        <td className="td max-w-56 truncate"><Link to={`/inspections/${r.id}`} className="hover:underline">{r.filename}</Link></td>
                        <td className="td"><DecisionBadge decision={r.decision} /></td>
                        <td className="td"><SeverityBadge severity={r.severity} /></td>
                        <td className="td text-right tabular-nums">{num(r.normalized_score, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}
    </>
  );
}
