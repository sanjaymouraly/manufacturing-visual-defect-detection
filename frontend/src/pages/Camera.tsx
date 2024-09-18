import { useQuery } from "@tanstack/react-query";
import { Camera as CamIcon, CameraOff, Pause, Play, Video } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { DecisionHeader } from "../components/InspectionResult";
import { ImageViewer } from "../components/inspection";
import ProductSelect from "../components/ProductSelect";
import { Button, Card, DecisionBadge, ErrorBox, Field, Note, PageHeader, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { ago, num } from "../lib/format";
import type { Inspection } from "../lib/types";

interface CamCfg { id: string; name: string; kind: string; source: string; product_key: string; enabled: boolean }
interface CameraSettings { default_interval_s: number; default_product: string; mirror_preview: boolean; resolution: string; cameras: CamCfg[] }

export default function CameraPage() {
  const settings = useQuery({ queryKey: ["settings"], queryFn: () => api<Record<string, any>>("/api/settings") });
  const camSet = settings.data?.camera as CameraSettings | undefined;
  const [mode, setMode] = useState<"browser" | "server">("browser");
  const [product, setProduct] = useState("");
  const [serverCam, setServerCam] = useState("");
  const [interval, setIntervalS] = useState(5);
  const [batch, setBatch] = useState("");
  const [auto, setAuto] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<unknown>(null);
  const [last, setLast] = useState<Inspection | null>(null);
  const [log, setLog] = useState<Inspection[]>([]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [streaming, setStreaming] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);

  useEffect(() => { if (camSet) setIntervalS(camSet.default_interval_s || 5); }, [camSet]);

  const stopStream = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    setStreaming(false);
    setAuto(false);
  }, []);
  useEffect(() => stopStream, [stopStream]);

  const startStream = async () => {
    setErr(null);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access requires a secure context (HTTPS or localhost) and a browser with getUserMedia support.");
      const [w, h] = (camSet?.resolution ?? "1280x720").split("x").map(Number);
      const s = await navigator.mediaDevices.getUserMedia({ video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: w }, height: { ideal: h } }, audio: false });
      stream.current = s;
      if (video.current) { video.current.srcObject = s; await video.current.play(); }
      setStreaming(true);
      setDevices((await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput"));
    } catch (e) {
      setErr(e instanceof Error ? new Error(`${e.name === "NotAllowedError" ? "Camera permission denied. " : e.name === "NotFoundError" ? "No camera found. " : ""}${e.message}`) : e);
    }
  };

  const captureBrowser = useCallback(async () => {
    const v = video.current;
    if (!v || !v.videoWidth) throw new Error("Camera stream not ready");
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    const blob = await new Promise<Blob>((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("Capture failed"))), "image/jpeg", 0.92));
    const fd = new FormData();
    fd.append("file", blob, `webcam-${new Date().toISOString().replace(/[:.]/g, "-")}.jpg`);
    fd.append("product_key", product);
    fd.append("source", "camera");
    fd.append("camera_id", "browser-webcam");
    fd.append("batch_code", batch);
    fd.append("force", "true");
    return api<Inspection>("/api/inspections", { form: fd });
  }, [product, batch]);

  const captureServer = useCallback(() => {
    const fd = new FormData();
    fd.append("product_key", product);
    fd.append("batch_code", batch);
    return api<Inspection>(`/api/cameras/${serverCam}/inspect`, { form: fd });
  }, [product, batch, serverCam]);

  const capture = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setErr(null);
    try {
      const r = mode === "browser" ? await captureBrowser() : await captureServer();
      setLast(r);
      setLog((l) => [r, ...l].slice(0, 50));
    } catch (e) {
      setErr(e);
      setAuto(false);
    } finally {
      setBusy(false);
    }
  }, [busy, mode, captureBrowser, captureServer]);

  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => { capture(); }, Math.max(1, interval) * 1000);
    return () => clearInterval(t);
  }, [auto, interval, capture]);

  const serverCams = (camSet?.cameras ?? []).filter((c) => c.enabled);
  const canCapture = !!product && (mode === "browser" ? streaming : !!serverCam);

  return (
    <>
      <PageHeader title="Camera inspection" description="Capture frames from a browser webcam or from a server-side camera adapter (USB/IP/HTTP snapshot/folder) and inspect them manually or at a fixed interval." />
      <Tabs tabs={[{ id: "browser", label: "Browser webcam" }, { id: "server", label: "Server camera adapters" }]} value={mode} onChange={(m) => { setAuto(false); setMode(m); }} />
      <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
        <div className="min-w-0 space-y-4">
          {mode === "browser" ? (
            <Card title="Live preview" actions={streaming ? <Button size="sm" onClick={stopStream}><CameraOff className="size-3.5" /> Stop camera</Button> : <Button size="sm" variant="primary" onClick={startStream}><Video className="size-3.5" /> Start camera</Button>}>
              <div className="relative aspect-video overflow-hidden rounded-lg bg-slate-900">
                <video ref={video} playsInline muted className="size-full object-contain" style={{ transform: camSet?.mirror_preview ? "scaleX(-1)" : undefined }} aria-label="Webcam preview" />
                {!streaming && <div className="absolute inset-0 grid place-items-center text-sm text-slate-400">Camera is off</div>}
                {auto && <span className="absolute top-2 left-2 flex items-center gap-1 rounded bg-red-600 px-2 py-0.5 text-xs font-semibold text-white"><span className="size-2 animate-pulse rounded-full bg-white" />AUTO {interval}s</span>}
              </div>
              {devices.length > 1 && (
                <div className="mt-2 max-w-sm"><Field label="Device">{(id) => <select id={id} className="input" value={deviceId} onChange={(e) => { setDeviceId(e.target.value); stopStream(); }}>{devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}</select>}</Field></div>
              )}
              <p className="muted mt-2 text-[11px]">Frames are sent only to this DefectLens server. Webcam access needs HTTPS or localhost.</p>
            </Card>
          ) : (
            <Card title="Server camera">
              {serverCams.length === 0 ? (
                <Note>No server cameras configured. An administrator can add USB, IP (RTSP/MJPEG), HTTP-snapshot or folder-watch cameras in <Link className="underline" to="/settings">Settings → Cameras</Link>.</Note>
              ) : (
                <Field label="Camera">{(id) => <select id={id} className="input max-w-md" value={serverCam} onChange={(e) => { setServerCam(e.target.value); const c = serverCams.find((x) => x.id === e.target.value); if (c?.product_key) setProduct(c.product_key); }}>
                  <option value="">Choose…</option>{serverCams.map((c) => <option key={c.id} value={c.id}>{c.name} ({c.kind})</option>)}</select>}</Field>
              )}
              {last?.images.display && mode === "server" && <img src={last.images.display} alt="Last captured frame" className="mt-3 max-h-80 rounded-lg" />}
            </Card>
          )}
          <ErrorBox error={err} title="Capture failed" />
          {last && (
            <div className="space-y-3">
              <DecisionHeader insp={last} />
              <Card title="Last frame"><ImageViewer insp={last} /></Card>
            </div>
          )}
        </div>
        <div className="space-y-4">
          <Card title="Controls">
            <div className="space-y-3">
              <Field label="Product">{(id) => <ProductSelect id={id} value={product} onChange={setProduct} requireActive />}</Field>
              <div className="grid grid-cols-2 gap-2">
                <Field label="Interval (seconds)">{(id) => <input id={id} type="number" min={1} max={3600} className="input" value={interval} onChange={(e) => setIntervalS(Number(e.target.value))} />}</Field>
                <Field label="Batch / lot">{(id) => <input id={id} className="input" value={batch} onChange={(e) => setBatch(e.target.value)} />}</Field>
              </div>
              <div className="flex gap-2">
                <Button variant="primary" className="flex-1" disabled={!canCapture} loading={busy && !auto} onClick={capture}><CamIcon className="size-4" /> Capture & inspect</Button>
                <Button className="flex-1" disabled={!canCapture} onClick={() => setAuto((a) => !a)} variant={auto ? "danger" : "secondary"}>{auto ? <><Pause className="size-4" /> Stop auto</> : <><Play className="size-4" /> Auto</>}</Button>
              </div>
            </div>
          </Card>
          <Card title={`Session snapshots (${log.length})`} bodyClass="p-0">
            {log.length === 0 ? <p className="muted p-4 text-sm">Captured frames appear here.</p> : (
              <ul className="max-h-[60vh] overflow-y-auto">
                {log.map((i) => (
                  <li key={i.id} className="flex items-center gap-3 border-b border-slate-100 px-3 py-2 text-sm dark:border-slate-800">
                    {i.images.display && <img src={i.images.overlay ?? i.images.display} alt="" className="h-10 w-14 rounded object-cover" />}
                    <DecisionBadge decision={i.decision} />
                    <span className="muted tabular-nums">{num(i.normalized_score, 2)}</span>
                    <Link to={`/inspections/${i.id}`} className="ml-auto text-xs underline">{ago(i.created_at)}</Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
