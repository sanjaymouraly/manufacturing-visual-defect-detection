import { useQuery } from "@tanstack/react-query";
import { ClipboardCheck } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import ProductSelect from "../components/ProductSelect";
import { Badge, Card, DecisionBadge, Empty, ErrorBox, PageHeader, SeverityBadge, Spinner } from "../components/ui";
import { api, qs } from "../lib/api";
import { ago, num, titleCase } from "../lib/format";
import type { Inspection } from "../lib/types";

const CAT_INFO: Record<string, string> = {
  inspection_failed: "The image could not be inspected (decode/model error). Never auto-passed.",
  borderline: "Score between the PASS and FAIL thresholds.",
  image_quality: "Image quality checks failed (blur, exposure, size…).",
  out_of_distribution: "Image looks unlike anything the model was trained on.",
  model_disagreement: "Primary and second-opinion model disagree.",
  high_risk: "Severe defect requiring inspector confirmation.",
  other: "Other review reasons.",
};

export default function ReviewQueue() {
  const [product, setProduct] = useState("");
  const [cat, setCat] = useState("");
  const nav = useNavigate();
  const q = useQuery({
    queryKey: ["queue", product, cat],
    queryFn: () => api<{ items: Inspection[]; total: number; categories: Record<string, number> }>(`/api/reviews/queue${qs({ product_key: product, category: cat })}`),
    refetchInterval: 10000,
  });
  return (
    <>
      <PageHeader title="Review queue" description="Inspections that need a human decision, ordered by risk: failed inspections first, then by severity and age."
        actions={<ProductSelect value={product} onChange={setProduct} allowAll className="input w-56" />} />
      <ErrorBox error={q.error} />
      {q.data && (
        <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by reason">
          <button onClick={() => setCat("")} className={`rounded-full border px-3 py-1 text-xs font-medium ${!cat ? "border-brand-600 bg-brand-700 text-white" : "border-slate-300 dark:border-slate-700"}`}>All ({q.data.total})</button>
          {Object.entries(q.data.categories).map(([c, n]) => (
            <button key={c} title={CAT_INFO[c]} onClick={() => setCat(c)} className={`rounded-full border px-3 py-1 text-xs font-medium ${cat === c ? "border-brand-600 bg-brand-700 text-white" : "border-slate-300 dark:border-slate-700"}`}>{titleCase(c)} ({n})</button>
          ))}
        </div>
      )}
      <Card bodyClass="p-0">
        {q.isLoading ? <Spinner /> : !q.data?.items.length ? (
          <Empty title="Queue is empty" icon={<ClipboardCheck className="size-8" />}>No inspections are waiting for review.</Empty>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Image</th><th className="th">Waiting</th><th className="th">Product</th><th className="th">Decision</th><th className="th">Severity</th><th className="th">Why</th><th className="th text-right">Norm. score</th></tr></thead>
              <tbody>
                {q.data.items.map((i) => (
                  <tr key={i.id} className="cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800/60 dark:hover:bg-slate-800/40" onClick={() => nav(`/reviews/${i.id}`)}>
                    <td className="td">{i.images.overlay || i.images.display ? <img src={(i.images.overlay ?? i.images.display)!} alt="" className="h-12 w-16 rounded object-cover" loading="lazy" /> : <div className="h-12 w-16 rounded bg-slate-200 dark:bg-slate-800" />}</td>
                    <td className="td muted whitespace-nowrap">{ago(i.created_at)}</td>
                    <td className="td">{i.product_key}<p className="muted max-w-48 truncate text-xs">{i.filename}</p></td>
                    <td className="td"><DecisionBadge decision={i.decision} /></td>
                    <td className="td"><SeverityBadge severity={i.severity} /></td>
                    <td className="td"><div className="flex flex-wrap gap-1">{(i.categories ?? []).map((c) => <Badge key={c} tone={c === "inspection_failed" ? "red" : "amber"} title={CAT_INFO[c]}>{titleCase(c)}</Badge>)}</div></td>
                    <td className="td text-right tabular-nums"><Link to={`/reviews/${i.id}`} onClick={(e) => e.stopPropagation()}>{num(i.normalized_score, 2)}</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
