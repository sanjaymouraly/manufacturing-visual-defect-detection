import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Check, Eraser, Square, X } from "lucide-react";
import { useEffect, useRef, useState, type PointerEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { DecisionHeader } from "../components/InspectionResult";
import { ScoreGauge, ProbabilityLine } from "../components/inspection";
import { Button, Card, ErrorBox, Field, Note, PageHeader, Spinner } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useToast } from "../lib/toast";
import type { Inspection, Severity } from "../lib/types";

type Box = [number, number, number, number];

/** Image with model heatmap and a drawing layer for inspector-marked defect regions (image pixel coordinates). */
function RegionEditor({ insp, boxes, setBoxes }: { insp: Inspection; boxes: Box[]; setBoxes: (b: Box[]) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<Box | null>(null);
  const start = useRef<[number, number] | null>(null);
  const [heat, setHeat] = useState(0.4);
  const w = insp.width ?? 1;
  const h = insp.height ?? 1;
  const toImg = (e: PointerEvent): [number, number] => {
    const r = ref.current!.getBoundingClientRect();
    return [Math.round(((e.clientX - r.left) / r.width) * w), Math.round(((e.clientY - r.top) / r.height) * h)];
  };
  const down = (e: PointerEvent) => { (e.target as Element).setPointerCapture(e.pointerId); start.current = toImg(e); setDraft(null); };
  const move = (e: PointerEvent) => {
    if (!start.current) return;
    const [x, y] = toImg(e);
    const [x0, y0] = start.current;
    setDraft([Math.max(0, Math.min(x, x0)), Math.max(0, Math.min(y, y0)), Math.abs(x - x0), Math.abs(y - y0)]);
  };
  const up = () => {
    if (draft && draft[2] > 3 && draft[3] > 3) setBoxes([...boxes, [draft[0], draft[1], Math.min(draft[2], w - draft[0]), Math.min(draft[3], h - draft[1])]]);
    start.current = null;
    setDraft(null);
  };
  if (!insp.images.display) return <p className="muted text-sm">Image not available (removed by retention or never stored). You can still record a label.</p>;
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-3 text-xs">
        <span className="flex items-center gap-1"><Square className="size-3.5" /> Drag on the image to mark defect regions</span>
        <label className="flex items-center gap-2"><span className="muted">Heatmap</span><input type="range" min={0} max={1} step={0.05} value={heat} onChange={(e) => setHeat(Number(e.target.value))} aria-label="Heatmap opacity" /></label>
        <Button size="sm" variant="ghost" disabled={!boxes.length} onClick={() => setBoxes(boxes.slice(0, -1))}><Eraser className="size-3.5" /> Undo</Button>
        <Button size="sm" variant="ghost" disabled={!boxes.length} onClick={() => setBoxes([])}>Clear</Button>
      </div>
      <div className="overflow-hidden rounded-lg bg-slate-900">
        <div ref={ref} className="relative mx-auto touch-none select-none" style={{ aspectRatio: `${w} / ${h}`, width: `min(100%, calc(65vh * ${w / h}))` }}
          onPointerDown={down} onPointerMove={move} onPointerUp={up} role="application" aria-label="Region drawing area">
          <img src={insp.images.display} alt="Inspected" className="absolute inset-0 size-full object-contain" draggable={false} />
          {insp.images.heatmap && <img src={insp.images.heatmap} alt="" className="pointer-events-none absolute inset-0 size-full object-contain" style={{ opacity: heat }} draggable={false} />}
          <svg className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${w} ${h}`} aria-hidden>
            {(insp.regions ?? []).map((r, i) => <rect key={`m${i}`} x={r.bbox[0]} y={r.bbox[1]} width={r.bbox[2]} height={r.bbox[3]} fill="none" stroke="#f43f5e" strokeWidth={Math.max(1.5, w / 400)} strokeOpacity={0.8} />)}
            {[...boxes, ...(draft ? [draft] : [])].map((b, i) => <rect key={i} x={b[0]} y={b[1]} width={b[2]} height={b[3]} fill="rgba(56,189,248,0.2)" stroke="#38bdf8" strokeWidth={Math.max(2, w / 300)} strokeDasharray="8 4" />)}
          </svg>
        </div>
      </div>
      <p className="muted mt-1 text-[11px]">Red: model regions · Blue dashed: your regions ({boxes.length}). Regions are saved with the review and used as masks if the sample is added to the feedback dataset.</p>
    </div>
  );
}

export default function ReviewDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const { can } = useAuth();
  const q = useQuery({ queryKey: ["inspection", id], queryFn: () => api<Inspection>(`/api/inspections/${id}`) });
  const queue = useQuery({ queryKey: ["queue", "", ""], queryFn: () => api<{ items: Inspection[] }>("/api/reviews/queue") });
  const [label, setLabel] = useState<"defective" | "normal" | "">("");
  const [defectType, setDefectType] = useState("");
  const [severity, setSeverity] = useState<Severity | "">("");
  const [comment, setComment] = useState("");
  const [feedback, setFeedback] = useState(true);
  const [boxes, setBoxes] = useState<Box[]>([]);

  useEffect(() => {
    setLabel(""); setDefectType(""); setSeverity(""); setComment(""); setBoxes([]); setFeedback(true);
  }, [id]);

  const nextId = queue.data?.items.find((i) => i.id !== id)?.id;
  const submit = useMutation({
    mutationFn: () => api(`/api/reviews/${id}`, { body: { label, defect_type: label === "defective" ? defectType : "", severity: label === "defective" ? severity : "none", comment, regions: label === "defective" ? boxes : [], add_to_feedback: feedback } }),
    onSuccess: () => {
      toast("ok", "Review saved");
      qc.invalidateQueries({ queryKey: ["queue"] });
      qc.invalidateQueries({ queryKey: ["queue-count"] });
      qc.invalidateQueries({ queryKey: ["inspection", id] });
      nav(nextId ? `/reviews/${nextId}` : "/reviews");
    },
  });

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, select")) return;
      if (e.key === "d") setLabel("defective");
      if (e.key === "n") setLabel("normal");
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const insp = q.data;
  return (
    <>
      <Link to="/reviews" className="muted mb-2 inline-flex items-center gap-1 text-sm hover:underline"><ArrowLeft className="size-4" /> Review queue</Link>
      <PageHeader title="Review inspection" description={insp ? `${insp.filename} · ${insp.product_key}` : undefined}
        actions={insp && <Link to={`/inspections/${insp.id}`} className="text-sm underline">Full details</Link>} />
      <ErrorBox error={q.error} />
      {q.isLoading && <Spinner />}
      {insp && (
        <div className="grid gap-5 xl:grid-cols-[1fr_400px]">
          <div className="min-w-0 space-y-4">
            <DecisionHeader insp={insp} />
            <Card title="Image"><RegionEditor insp={insp} boxes={boxes} setBoxes={setBoxes} /></Card>
          </div>
          <div className="space-y-4">
            <Card title="Model evidence">
              <ScoreGauge score={insp.score} pass={insp.pass_threshold} fail={insp.fail_threshold} />
              <div className="mt-3"><ProbabilityLine insp={insp} /></div>
            </Card>
            <Card title="Your decision" subtitle="Shortcuts: D = defective, N = normal">
              {!can("admin", "inspector") ? <Note>Viewers cannot submit reviews.</Note> : (
                <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); submit.mutate(); }}>
                  <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Label">
                    <button type="button" role="radio" aria-checked={label === "defective"} onClick={() => setLabel("defective")} className={`flex items-center justify-center gap-2 rounded-lg border-2 p-3 text-sm font-semibold ${label === "defective" ? "border-red-500 bg-red-50 text-red-700 dark:bg-red-950" : "border-slate-200 dark:border-slate-700"}`}><X className="size-4" /> Defective</button>
                    <button type="button" role="radio" aria-checked={label === "normal"} onClick={() => setLabel("normal")} className={`flex items-center justify-center gap-2 rounded-lg border-2 p-3 text-sm font-semibold ${label === "normal" ? "border-emerald-500 bg-emerald-50 text-emerald-700 dark:bg-emerald-950" : "border-slate-200 dark:border-slate-700"}`}><Check className="size-4" /> Normal</button>
                  </div>
                  {label === "defective" && (
                    <>
                      <Field label="Defect type" hint="Free text, e.g. scratch, missing component, contamination">{(fid) => <input id={fid} className="input" value={defectType} onChange={(e) => setDefectType(e.target.value)} maxLength={64} list="defect-types" />}</Field>
                      <datalist id="defect-types">{["scratch", "crack", "dent", "contamination", "missing_component", "misalignment", "discoloration", "burr", "hole", "bent", "melt"].map((d) => <option key={d} value={d} />)}</datalist>
                      <Field label="Severity">{(fid) => <select id={fid} className="input" required value={severity} onChange={(e) => setSeverity(e.target.value as Severity)}><option value="" disabled>Choose severity…</option>{["low", "medium", "high", "critical"].map((s) => <option key={s} value={s}>{s}</option>)}</select>}</Field>
                    </>
                  )}
                  <Field label="Comment">{(fid) => <textarea id={fid} className="input min-h-20" value={comment} onChange={(e) => setComment(e.target.value)} maxLength={2000} />}</Field>
                  <label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-1" checked={feedback} onChange={(e) => setFeedback(e.target.checked)} />
                    <span>Propose for the feedback dataset<span className="muted block text-xs">An administrator must approve it before it can be used for retraining. Models are never retrained automatically.</span></span></label>
                  <ErrorBox error={submit.error} title="Review not saved" />
                  <Button variant="primary" className="w-full" disabled={!label || (label === "defective" && !severity)} loading={submit.isPending}>Submit review{nextId ? " & next" : ""}</Button>
                  {insp.reviews && insp.reviews.length > 0 && <p className="muted text-xs">This inspection already has {insp.reviews.length} review(s); a new review is appended and becomes the final label.</p>}
                </form>
              )}
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
