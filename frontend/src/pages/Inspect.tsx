import { useMutation } from "@tanstack/react-query";
import { ImagePlus, Loader2, RotateCcw, ScanSearch } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import InspectionResult from "../components/InspectionResult";
import ProductSelect from "../components/ProductSelect";
import { Button, Card, ErrorBox, Field, Note, PageHeader } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useModels } from "../lib/hooks";
import type { Inspection } from "../lib/types";

export default function Inspect() {
  const [product, setProduct] = useState("");
  const [modelId, setModelId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [batch, setBatch] = useState("");
  const [station, setStation] = useState("");
  const [ref, setRef] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const { can } = useAuth();
  const models = useModels(product);
  const approved = (models.data ?? []).filter((m) => m.status === "approved");

  const pick = useCallback((f: File | null | undefined) => {
    if (!f) return;
    setFile(f);
    setPreview((old) => { if (old) URL.revokeObjectURL(old); return URL.createObjectURL(f); });
    run.reset();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) pick(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [pick]);

  const run = useMutation({
    mutationFn: (force: boolean) => {
      const fd = new FormData();
      fd.append("file", file!);
      fd.append("product_key", product);
      if (modelId) fd.append("model_id", modelId);
      fd.append("batch_code", batch);
      fd.append("station", station);
      fd.append("external_ref", ref);
      fd.append("force", String(force));
      return api<Inspection>("/api/inspections", { form: fd });
    },
  });
  const onDrop = (e: DragEvent) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); };

  return (
    <>
      <PageHeader title="Inspect" description="Inspect a single image with the active model of a product. Supported: PNG, JPEG, BMP, TIFF, WebP. You can also paste an image (Ctrl+V)." />
      <div className="grid gap-5 lg:grid-cols-[360px_1fr]">
        <Card title="Input">
          <div className="space-y-3">
            <Field label="Product">{(id) => <ProductSelect id={id} value={product} onChange={(v) => { setProduct(v); setModelId(""); }} />}</Field>
            {can("admin") && (
              <Field label="Model" hint="Defaults to the product's active model. Only approved models can be selected.">
                {(id) => (
                  <select id={id} className="input" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                    <option value="">Active model</option>
                    {approved.map((m) => <option key={m.id} value={m.id}>{m.name} (v{m.version}, {m.algorithm})</option>)}
                  </select>
                )}
              </Field>
            )}
            <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
              className={`flex min-h-44 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-3 text-center text-sm transition-colors ${drag ? "border-brand-500 bg-brand-50 dark:bg-brand-800/20" : "border-slate-300 dark:border-slate-700"}`}
              onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()} aria-label="Choose or drop an image">
              {preview ? <img src={preview} alt="Selected" className="max-h-48 rounded object-contain" /> : <><ImagePlus className="size-8 text-slate-400" /><span>Drop an image here or <span className="font-medium text-brand-700 underline dark:text-brand-500">browse</span></span></>}
              {file && <span className="muted text-xs">{file.name} · {(file.size / 1024).toFixed(0)} KB</span>}
            </div>
            <input ref={input} type="file" accept="image/*,.tif,.tiff,.bmp" className="hidden" onChange={(e) => pick(e.target.files?.[0])} data-testid="file-input" />
            <div className="grid grid-cols-2 gap-2">
              <Field label="Batch / lot">{(id) => <input id={id} className="input" value={batch} onChange={(e) => setBatch(e.target.value)} maxLength={128} />}</Field>
              <Field label="Station">{(id) => <input id={id} className="input" value={station} onChange={(e) => setStation(e.target.value)} maxLength={64} />}</Field>
            </div>
            <Field label="Unit / serial reference (optional)">{(id) => <input id={id} className="input" value={ref} onChange={(e) => setRef(e.target.value)} maxLength={128} />}</Field>
            <Button variant="primary" className="w-full" disabled={!file || !product} loading={run.isPending} onClick={() => run.mutate(false)}><ScanSearch className="size-4" /> Inspect</Button>
            {file && <Button variant="ghost" className="w-full" onClick={() => { setFile(null); setPreview(null); run.reset(); }}><RotateCcw className="size-4" /> Clear</Button>}
          </div>
        </Card>
        <div className="min-w-0">
          <ErrorBox error={run.error} title="Inspection was not performed" />
          {run.isPending && <div className="card muted flex items-center gap-2 p-6 text-sm"><Loader2 className="size-4 animate-spin" /> Running inference…</div>}
          {run.data && (
            <div className="space-y-3">
              {run.data.duplicate && (
                <Note>This exact image was already inspected with the same model, so the existing result is shown (no duplicate record created).
                  <Button size="sm" className="ml-2" onClick={() => run.mutate(true)}>Inspect again anyway</Button></Note>
              )}
              <InspectionResult insp={run.data} />
            </div>
          )}
          {!run.data && !run.isPending && !run.error && (
            <div className="card flex h-full min-h-64 flex-col items-center justify-center gap-2 p-8 text-center">
              <ScanSearch className="size-10 text-slate-300" />
              <p className="font-medium">No result yet</p>
              <p className="muted max-w-md text-sm">Choose a product and an image, then press Inspect. The result shows the decision (PASS / FAIL / REVIEW), its reasons, the anomaly heatmap and localized regions.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
