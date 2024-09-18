import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Camera, CheckCircle2, KeyRound, Plug, Plus, Save, ShieldCheck, Trash2, XCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { JobStatus } from "../components/inspection";
import { Badge, Button, Card, ConfirmButton, ErrorBox, Field, KV, Modal, Note, PageHeader, Spinner, Tabs, Toggle } from "../components/ui";
import { api, qs } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, dt, num, titleCase } from "../lib/format";
import { useModels, useProducts } from "../lib/hooks";
import { useToast } from "../lib/toast";
import type { Job, Product, Provider, User } from "../lib/types";

type AnyRec = Record<string, any>;
type TabId = "policy" | "products" | "cameras" | "users" | "providers" | "privacy" | "audit" | "system";

function useSettings() {
  return useQuery({ queryKey: ["settings"], queryFn: () => api<Record<string, AnyRec>>("/api/settings") });
}

/** Local editable copy of one settings section with a Save button that PUTs it. */
function useSection(key: string) {
  const s = useSettings();
  const qc = useQueryClient();
  const toast = useToast();
  const [val, setVal] = useState<AnyRec | null>(null);
  useEffect(() => { if (s.data && val === null) setVal(structuredClone(s.data[key])); }, [s.data, key, val]);
  const save = useMutation({
    mutationFn: (v: AnyRec) => api<AnyRec>(`/api/settings/${key}`, { method: "PUT", body: v }),
    onSuccess: (v) => { setVal(v); qc.invalidateQueries({ queryKey: ["settings"] }); toast("ok", `${titleCase(key)} settings saved`); },
  });
  const dirty = !!s.data && !!val && JSON.stringify(s.data[key]) !== JSON.stringify(val);
  const set = (k: string, v: unknown) => setVal((x) => ({ ...(x ?? {}), [k]: v }));
  return { val, set, setVal, save, dirty, loading: s.isLoading, error: s.error };
}

function SaveBar({ dirty, save, onSave }: { dirty: boolean; save: { isPending: boolean; error: unknown }; onSave: () => void }) {
  return (
    <div className="mt-4 flex items-center gap-3">
      <Button variant="primary" disabled={!dirty} loading={save.isPending} onClick={onSave}><Save className="size-4" /> Save</Button>
      {dirty && <span className="text-xs text-amber-600">Unsaved changes</span>}
      <ErrorBox error={save.error} />
    </div>
  );
}

function NumField({ label, value, onChange, step = 1, min, max, hint }: { label: string; value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; hint?: ReactNode }) {
  return <Field label={label} hint={hint}>{(id) => <input id={id} type="number" className="input" value={value} step={step} min={min} max={max} onChange={(e) => onChange(Number(e.target.value))} />}</Field>;
}

// ---------------------------------------------------------------- policy, thresholds & quality
function PolicyTab() {
  const pol = useSection("policy");
  const qual = useSection("quality");
  const gen = useSection("general");
  const products = useProducts();
  if (!pol.val || !qual.val || !gen.val) return <Spinner />;
  const p = pol.val, q = qual.val, g = gen.val;
  const sev = ["low", "medium", "high", "critical"];
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <Card title="Decision policy" subtitle="Applies to every product unless a product overrides it. The decision rule itself is fixed and conservative: FAIL ≥ reject threshold, PASS only below the pass threshold with no flags, everything else REVIEW; failed inspections always go to REVIEW.">
        <div className="space-y-3">
          <Toggle checked={p.review_on_quality_issues} onChange={(v) => pol.set("review_on_quality_issues", v)} label="Route images with quality problems (dark, blurred, saturated…) to REVIEW instead of PASS" />
          <Toggle checked={p.review_on_ood} onChange={(v) => pol.set("review_on_ood", v)} label="Route out-of-distribution images (unlike the training data) to REVIEW" />
          <Toggle checked={p.review_on_disagreement} onChange={(v) => pol.set("review_on_disagreement", v)} label="Route to REVIEW when the secondary (shadow) model disagrees" />
          <fieldset><legend className="label">Also queue FAIL decisions of these severities for human confirmation</legend>
            <div className="flex gap-3">{sev.map((s) => <label key={s} className="flex items-center gap-1 text-sm"><input type="checkbox" checked={p.review_fail_severities.includes(s)} onChange={(e) => pol.set("review_fail_severities", e.target.checked ? [...p.review_fail_severities, s] : p.review_fail_severities.filter((x: string) => x !== s))} />{s}</label>)}</div></fieldset>
          <div className="grid grid-cols-3 gap-3">
            <NumField label="Medium ≥ score ratio" step={0.1} value={p.medium_ratio} onChange={(v) => pol.set("medium_ratio", v)} />
            <NumField label="High ≥ score ratio" step={0.1} value={p.high_ratio} onChange={(v) => pol.set("high_ratio", v)} />
            <NumField label="Critical ≥ score ratio" step={0.1} value={p.critical_ratio} onChange={(v) => pol.set("critical_ratio", v)} />
            <NumField label="High ≥ defect area" step={0.01} value={p.high_area_fraction} onChange={(v) => pol.set("high_area_fraction", v)} />
            <NumField label="Critical ≥ defect area" step={0.01} value={p.critical_area_fraction} onChange={(v) => pol.set("critical_area_fraction", v)} />
          </div>
          <p className="muted text-xs">Score ratio = raw anomaly score ÷ the model's reject threshold. Area = fraction of the image covered by detected regions.</p>
        </div>
        <SaveBar dirty={pol.dirty} save={pol.save} onSave={() => pol.save.mutate(p)} />
      </Card>
      <div className="space-y-5">
        <Card title="Per-model thresholds" subtitle="Thresholds belong to each model version (selected on its validation split). Edit them on the model page; every change needs a reason and is audited.">
          <table className="w-full text-sm"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Product</th><th className="th">Active model</th><th className="th text-right">Pass &lt;</th><th className="th text-right">Reject ≥</th><th className="th"></th></tr></thead>
            <tbody>{(products.data ?? []).map((pr) => (
              <tr key={pr.key} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td">{pr.name}</td><td className="td">{pr.active_model ? `${pr.active_model.name} v${pr.active_model.version}` : <span className="muted">none</span>}</td>
                <td className="td text-right tabular-nums">{num(pr.active_model?.thresholds.pass_threshold ?? null)}</td><td className="td text-right tabular-nums">{num(pr.active_model?.thresholds.fail_threshold ?? null)}</td>
                <td className="td">{pr.active_model && <Link className="text-brand-700 underline dark:text-brand-400" to={`/models/${pr.active_model.id}?tab=thresholds`}>Edit</Link>}</td></tr>
            ))}</tbody></table>
        </Card>
        <Card title="Image quality gate" subtitle="Images outside these limits are flagged; with the policy above they can never auto-PASS.">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            <NumField label="Min side (px)" value={q.min_side} onChange={(v) => qual.set("min_side", v)} />
            <NumField label="Min brightness (0–255)" value={q.min_brightness} onChange={(v) => qual.set("min_brightness", v)} />
            <NumField label="Max brightness (0–255)" value={q.max_brightness} onChange={(v) => qual.set("max_brightness", v)} />
            <NumField label="Min contrast (std)" value={q.min_contrast} onChange={(v) => qual.set("min_contrast", v)} />
            <NumField label="Min sharpness (Laplacian var.)" value={q.min_sharpness} onChange={(v) => qual.set("min_sharpness", v)} />
            <NumField label="Max saturated fraction" step={0.01} value={q.max_saturated_fraction} onChange={(v) => qual.set("max_saturated_fraction", v)} />
          </div>
          <SaveBar dirty={qual.dirty} save={qual.save} onSave={() => qual.save.mutate(q)} />
        </Card>
        <Card title="General">
          <div className="grid gap-3 md:grid-cols-2">
            <Field label="Site name">{(id) => <input id={id} className="input" value={g.site_name} onChange={(e) => gen.set("site_name", e.target.value)} />}</Field>
            <NumField label="Duplicate-upload window (hours)" value={g.duplicate_window_hours} min={0} onChange={(v) => gen.set("duplicate_window_hours", v)} hint="Same image + product inside this window returns the earlier result unless forced." />
            <Field label="Operating mode" hint="Decision support: humans own final decisions. Automated gating should only be enabled after site validation.">{(id) => <select id={id} className="input" value={g.operating_mode} onChange={(e) => gen.set("operating_mode", e.target.value)}><option value="decision_support">Decision support (recommended)</option><option value="automated_gate">Automated gate (validated lines only)</option></select>}</Field>
          </div>
          <SaveBar dirty={gen.dirty} save={gen.save} onSave={() => gen.save.mutate(g)} />
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- products
function ProductRow({ p }: { p: Product }) {
  const qc = useQueryClient();
  const toast = useToast();
  const models = useModels(p.key);
  const [name, setName] = useState(p.name);
  const [desc, setDesc] = useState(p.description);
  const [secondary, setSecondary] = useState(p.secondary_model_id ? String(p.secondary_model_id) : "");
  const save = useMutation({
    mutationFn: () => api(`/api/products/${p.key}`, { method: "PATCH", body: { name, description: desc, ...(secondary ? { secondary_model_id: Number(secondary) } : { clear_secondary: true }) } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["products"] }); toast("ok", `${p.key} updated`); },
  });
  const approved = (models.data ?? []).filter((m) => m.status === "approved" && m.id !== p.active_model_id);
  const dirty = name !== p.name || desc !== p.description || secondary !== (p.secondary_model_id ? String(p.secondary_model_id) : "");
  return (
    <tr className="border-b border-slate-100 align-top dark:border-slate-800/60">
      <td className="td font-mono text-xs">{p.key}{p.synthetic && <div><Badge tone="violet">synthetic</Badge></div>}</td>
      <td className="td"><input className="input" value={name} onChange={(e) => setName(e.target.value)} aria-label={`${p.key} name`} /><textarea className="input mt-1 text-xs" rows={2} value={desc} onChange={(e) => setDesc(e.target.value)} aria-label={`${p.key} description`} /></td>
      <td className="td text-sm">{p.active_model ? <Link className="underline" to={`/models/${p.active_model.id}`}>{p.active_model.name} v{p.active_model.version}</Link> : <span className="muted">none</span>}<p className="muted text-xs">Change in Model lab</p></td>
      <td className="td"><select className="input" value={secondary} onChange={(e) => setSecondary(e.target.value)} aria-label={`${p.key} secondary model`}><option value="">none</option>{approved.map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version}</option>)}</select><p className="muted mt-1 text-xs">Shadow model; disagreement → REVIEW</p></td>
      <td className="td"><Button size="sm" variant="primary" disabled={!dirty} loading={save.isPending} onClick={() => save.mutate()}><Save className="size-3.5" /> Save</Button><ErrorBox error={save.error} /></td>
    </tr>
  );
}

function ProductsTab() {
  const products = useProducts();
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ key: "", name: "", description: "" });
  const create = useMutation({ mutationFn: () => api("/api/products", { body: form }), onSuccess: () => { qc.invalidateQueries({ queryKey: ["products"] }); setOpen(false); setForm({ key: "", name: "", description: "" }); } });
  return (
    <Card title="Products" subtitle="Each product (part type / inspection station) has its own active model, thresholds and optional shadow model." actions={<Button size="sm" onClick={() => setOpen(true)}><Plus className="size-4" /> New product</Button>} bodyClass="p-0">
      <div className="overflow-x-auto"><table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Key</th><th className="th">Name / description</th><th className="th">Active model</th><th className="th">Secondary model</th><th className="th"></th></tr></thead>
        <tbody>{(products.data ?? []).map((p) => <ProductRow key={p.key + p.name + p.secondary_model_id} p={p} />)}</tbody></table></div>
      <Modal open={open} onClose={() => setOpen(false)} title="New product">
        <div className="space-y-3">
          <Field label="Key (lowercase, a-z 0-9 _ -)">{(id) => <input id={id} className="input" value={form.key} onChange={(e) => setForm({ ...form, key: e.target.value })} />}</Field>
          <Field label="Name">{(id) => <input id={id} className="input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />}</Field>
          <Field label="Description">{(id) => <textarea id={id} className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />}</Field>
          <Note>A new product has no model. Import a dataset for it (Datasets), train a candidate (Model lab), review its held-out metrics, approve and activate it.</Note>
          <ErrorBox error={create.error} />
          <Button variant="primary" disabled={!/^[a-z0-9][a-z0-9_-]{1,63}$/.test(form.key) || !form.name} loading={create.isPending} onClick={() => create.mutate()}>Create</Button>
        </div>
      </Modal>
    </Card>
  );
}

// ---------------------------------------------------------------- cameras
interface Cam { id: string; name: string; kind: "usb" | "ip" | "http_snapshot" | "folder"; source: string; product_key: string; enabled: boolean }
const CAM_HINT: Record<Cam["kind"], string> = { usb: "Device index, e.g. 0", ip: "rtsp://user:pass@host:554/stream or http://host/mjpeg", http_snapshot: "http://camera/snapshot.jpg", folder: "Absolute folder path watched for new images (inside the configured camera root)" };

function CamerasTab() {
  const sec = useSection("camera");
  const products = useProducts();
  const toast = useToast();
  const [tests, setTests] = useState<Record<string, { ok: boolean; error?: string; bytes?: number }>>({});
  const test = useMutation({ mutationFn: (id: string) => api<{ ok: boolean; error?: string; bytes?: number }>(`/api/cameras/${id}/test`, { method: "POST" }), onSuccess: (r, id) => setTests((t) => ({ ...t, [id]: r })), onError: (e: Error) => toast("error", e.message) });
  const savedCams = useSettingsCache();
  if (!sec.val) return <Spinner />;
  const v = sec.val;
  const cams: Cam[] = v.cameras;
  const upd = (i: number, patch: Partial<Cam>) => sec.set("cameras", cams.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const saved = new Set<string>((savedCams ?? []).map((c: Cam) => c.id));
  return (
    <div className="space-y-5">
      <Card title="Browser camera defaults" subtitle="Used by the Camera page for webcams attached to the operator's computer.">
        <div className="grid gap-3 md:grid-cols-4">
          <NumField label="Default capture interval (s)" min={1} max={3600} value={v.default_interval_s} onChange={(x) => sec.set("default_interval_s", x)} />
          <Field label="Default product">{(id) => <select id={id} className="input" value={v.default_product} onChange={(e) => sec.set("default_product", e.target.value)}><option value="">(none)</option>{(products.data ?? []).map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}</select>}</Field>
          <Field label="Requested resolution">{(id) => <select id={id} className="input" value={v.resolution} onChange={(e) => sec.set("resolution", e.target.value)}>{["640x480", "1280x720", "1920x1080"].map((r) => <option key={r}>{r}</option>)}</select>}</Field>
          <div className="pt-6"><Toggle checked={v.mirror_preview} onChange={(x) => sec.set("mirror_preview", x)} label="Mirror preview" /></div>
        </div>
      </Card>
      <Card title="Server-side cameras" subtitle="Cameras the server can capture from directly (USB on the server, IP/RTSP streams, HTTP snapshot URLs, watched folders)." actions={<Button size="sm" onClick={() => sec.set("cameras", [...cams, { id: `cam${cams.length + 1}`, name: `Camera ${cams.length + 1}`, kind: "http_snapshot", source: "", product_key: "", enabled: true }])}><Plus className="size-4" /> Add camera</Button>}>
        {!cams.length ? <p className="muted text-sm">No server cameras configured.</p> : (
          <div className="space-y-3">{cams.map((c, i) => (
            <div key={i} className="grid gap-2 rounded-lg border border-slate-200 p-3 md:grid-cols-[110px_1fr_150px_2fr_160px_auto] dark:border-slate-800">
              <input className="input font-mono text-xs" value={c.id} onChange={(e) => upd(i, { id: e.target.value })} aria-label="Camera id" />
              <input className="input" value={c.name} onChange={(e) => upd(i, { name: e.target.value })} aria-label="Camera name" />
              <select className="input" value={c.kind} onChange={(e) => upd(i, { kind: e.target.value as Cam["kind"] })} aria-label="Kind"><option value="usb">USB</option><option value="ip">IP (RTSP/MJPEG)</option><option value="http_snapshot">HTTP snapshot</option><option value="folder">Watched folder</option></select>
              <input className="input text-xs" value={c.source} placeholder={CAM_HINT[c.kind]} onChange={(e) => upd(i, { source: e.target.value })} aria-label="Source" />
              <select className="input" value={c.product_key} onChange={(e) => upd(i, { product_key: e.target.value })} aria-label="Product"><option value="">product…</option>{(products.data ?? []).map((p) => <option key={p.key} value={p.key}>{p.name}</option>)}</select>
              <div className="flex items-center gap-1">
                <Toggle checked={c.enabled} onChange={(x) => upd(i, { enabled: x })} label="On" />
                <Button size="sm" variant="ghost" disabled={!saved.has(c.id) || sec.dirty} title={sec.dirty ? "Save first" : "Capture one frame"} onClick={() => test.mutate(c.id)}><Camera className="size-3.5" /> Test</Button>
                <Button size="sm" variant="ghost" onClick={() => sec.set("cameras", cams.filter((_, j) => j !== i))} aria-label="Remove camera"><Trash2 className="size-3.5" /></Button>
              </div>
              {tests[c.id] && <p className={`text-xs md:col-span-6 ${tests[c.id].ok ? "text-emerald-600" : "text-red-600"}`}>{tests[c.id].ok ? `OK: captured ${tests[c.id].bytes} bytes` : `Failed: ${tests[c.id].error}`}</p>}
            </div>
          ))}</div>
        )}
        <Note>Physical USB/IP cameras could not be tested in the development environment; the folder and HTTP-snapshot adapters are covered by automated tests.</Note>
      </Card>
      <SaveBar dirty={sec.dirty} save={sec.save} onSave={() => sec.save.mutate(v)} />
    </div>
  );
}
function useSettingsCache() { const s = useSettings(); return s.data?.camera?.cameras as Cam[] | undefined; }

// ---------------------------------------------------------------- users
function UsersTab() {
  const { user: me } = useAuth();
  const qc = useQueryClient();
  const toast = useToast();
  const users = useQuery({ queryKey: ["users"], queryFn: () => api<User[]>("/api/users") });
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState({ username: "", full_name: "", password: "", role: "inspector" });
  const [pwFor, setPwFor] = useState<User | null>(null);
  const [newPw, setNewPw] = useState("");
  const patch = useMutation({ mutationFn: ({ id, body }: { id: number; body: AnyRec }) => api(`/api/users/${id}`, { method: "PATCH", body }), onSuccess: () => { qc.invalidateQueries({ queryKey: ["users"] }); toast("ok", "User updated (existing sessions of that user were revoked where relevant)"); }, onError: (e: Error) => toast("error", e.message) });
  const create = useMutation({ mutationFn: () => api("/api/users", { body: form }), onSuccess: () => { qc.invalidateQueries({ queryKey: ["users"] }); setOpen(false); setForm({ username: "", full_name: "", password: "", role: "inspector" }); } });
  return (
    <Card title="Users & roles" subtitle="admin: everything · inspector: inspect, review, batch, camera · viewer: read-only" actions={<Button size="sm" onClick={() => setOpen(true)}><Plus className="size-4" /> New user</Button>} bodyClass="p-0">
      <table className="w-full"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">User</th><th className="th">Role</th><th className="th">Status</th><th className="th">Last login</th><th className="th"></th></tr></thead>
        <tbody>{(users.data ?? []).map((u) => (
          <tr key={u.id} className="border-b border-slate-100 dark:border-slate-800/60">
            <td className="td"><b>{u.username}</b>{u.id === me?.id && <span className="muted"> (you)</span>}<p className="muted text-xs">{u.full_name}</p></td>
            <td className="td"><select className="input w-32 py-1" value={u.role} disabled={u.id === me?.id} onChange={(e) => patch.mutate({ id: u.id, body: { role: e.target.value } })} aria-label={`${u.username} role`}><option>admin</option><option>inspector</option><option>viewer</option></select></td>
            <td className="td">{u.is_active ? <Badge tone="green">active</Badge> : <Badge tone="slate">disabled</Badge>}</td>
            <td className="td muted text-xs">{u.last_login_at ? ago(u.last_login_at) : "never"}</td>
            <td className="td"><div className="flex gap-1">
              <Button size="sm" variant="ghost" onClick={() => { setPwFor(u); setNewPw(""); }}><KeyRound className="size-3.5" /> Reset password</Button>
              {u.id !== me?.id && <ConfirmButton size="sm" variant={u.is_active ? "danger" : "secondary"} confirmText={u.is_active ? `Disable ${u.username}? Their sessions end immediately.` : `Re-enable ${u.username}?`} onConfirm={() => patch.mutate({ id: u.id, body: { is_active: !u.is_active } })}>{u.is_active ? "Disable" : "Enable"}</ConfirmButton>}
            </div></td>
          </tr>
        ))}</tbody></table>
      <Modal open={open} onClose={() => setOpen(false)} title="New user">
        <div className="space-y-3">
          <Field label="Username">{(id) => <input id={id} className="input" autoComplete="off" value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />}</Field>
          <Field label="Full name">{(id) => <input id={id} className="input" value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} />}</Field>
          <Field label="Initial password" hint="At least 8 characters; common passwords are rejected.">{(id) => <input id={id} type="password" autoComplete="new-password" className="input" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />}</Field>
          <Field label="Role">{(id) => <select id={id} className="input" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option>inspector</option><option>viewer</option><option>admin</option></select>}</Field>
          <ErrorBox error={create.error} />
          <Button variant="primary" disabled={form.username.length < 3 || form.password.length < 8} loading={create.isPending} onClick={() => create.mutate()}>Create user</Button>
        </div>
      </Modal>
      <Modal open={!!pwFor} onClose={() => setPwFor(null)} title={`Reset password: ${pwFor?.username ?? ""}`}>
        <div className="space-y-3">
          <Field label="New password">{(id) => <input id={id} type="password" autoComplete="new-password" className="input" value={newPw} onChange={(e) => setNewPw(e.target.value)} />}</Field>
          <p className="muted text-xs">All existing sessions of this user are revoked.</p>
          <Button variant="primary" disabled={newPw.length < 8} loading={patch.isPending} onClick={() => { patch.mutate({ id: pwFor!.id, body: { password: newPw } }); setPwFor(null); }}>Set password</Button>
        </div>
      </Modal>
    </Card>
  );
}

// ---------------------------------------------------------------- AI providers
interface ProvList { providers: Provider[]; kinds: string[]; default_base_urls: Record<string, string>; active_provider_id: number | null; usage: { provider: string; calls: number; tokens_in: number; tokens_out: number; cost_usd: number | null }[] }
const KIND_LABEL: Record<string, string> = { openai: "OpenAI", anthropic: "Anthropic", openrouter: "OpenRouter", openai_compatible: "Custom OpenAI-compatible", ollama: "Ollama (local)" };
const emptyProv = { name: "", kind: "ollama", base_url: "", model: "", api_key: "", api_key_env: "", timeout_s: 30, max_retries: 2, max_tokens: 600, temperature: 0.2, price_in_per_mtok: null as number | null, price_out_per_mtok: null as number | null, enabled: true };

function ProviderForm({ initial, kinds, defaults, onDone }: { initial: Provider | null; kinds: string[]; defaults: Record<string, string>; onDone: () => void }) {
  const qc = useQueryClient();
  const [f, setF] = useState(() => initial ? { ...emptyProv, ...initial, api_key: "" } : { ...emptyProv });
  const [keyTouched, setKeyTouched] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const fetchModels = useMutation({ mutationFn: () => api<{ models: string[] }>(`/api/providers/${initial!.id}/models`), onSuccess: (r) => setModels(r.models) });
  const save = useMutation({
    mutationFn: () => {
      const body: AnyRec = { name: f.name, kind: f.kind, base_url: f.base_url, model: f.model, api_key_env: f.api_key_env, timeout_s: f.timeout_s, max_retries: f.max_retries, max_tokens: f.max_tokens, temperature: f.temperature, price_in_per_mtok: f.price_in_per_mtok, price_out_per_mtok: f.price_out_per_mtok, enabled: f.enabled };
      if (!initial || keyTouched) body.api_key = f.api_key;
      return initial ? api(`/api/providers/${initial.id}`, { method: "PUT", body }) : api("/api/providers", { body });
    },
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["providers"] }); onDone(); },
  });
  const set = (k: string, v: unknown) => setF((x) => ({ ...x, [k]: v }));
  const optNum = (v: string) => (v === "" ? null : Number(v));
  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Display name">{(id) => <input id={id} className="input" value={f.name} onChange={(e) => set("name", e.target.value)} />}</Field>
        <Field label="Provider type">{(id) => <select id={id} className="input" value={f.kind} onChange={(e) => set("kind", e.target.value)}>{kinds.map((k) => <option key={k} value={k}>{KIND_LABEL[k] ?? k}</option>)}</select>}</Field>
        <Field label="Base URL" hint={f.kind === "openai_compatible" ? "Required, e.g. http://localhost:8080/v1" : `Default: ${defaults[f.kind] ?? ""}`}>{(id) => <input id={id} className="input" value={f.base_url} placeholder={defaults[f.kind] ?? ""} onChange={(e) => set("base_url", e.target.value)} />}</Field>
        <Field label="Model">{(id) => <div className="flex gap-1"><input id={id} className="input" list="prov-models" value={f.model} onChange={(e) => set("model", e.target.value)} />{initial && <Button size="sm" onClick={() => fetchModels.mutate()} loading={fetchModels.isPending}>List</Button>}<datalist id="prov-models">{models.map((m) => <option key={m} value={m} />)}</datalist></div>}</Field>
        <Field label={initial?.api_key_set ? `API key (stored: ${initial.api_key_masked}; leave blank to keep)` : "API key"} hint="Encrypted at rest on the server and never sent back to the browser.">{(id) => <input id={id} type="password" autoComplete="off" className="input" value={f.api_key} onChange={(e) => { set("api_key", e.target.value); setKeyTouched(true); }} />}</Field>
        <Field label="…or read key from environment variable" hint="e.g. OPENAI_API_KEY (takes precedence when set)">{(id) => <input id={id} className="input font-mono" value={f.api_key_env} onChange={(e) => set("api_key_env", e.target.value.toUpperCase())} />}</Field>
      </div>
      <div className="grid gap-3 md:grid-cols-6">
        <NumField label="Timeout (s)" value={f.timeout_s} min={1} max={300} onChange={(v) => set("timeout_s", v)} />
        <NumField label="Retries" value={f.max_retries} min={0} max={6} onChange={(v) => set("max_retries", v)} />
        <NumField label="Max tokens" value={f.max_tokens} min={16} max={8192} onChange={(v) => set("max_tokens", v)} />
        <NumField label="Temperature" value={f.temperature} step={0.1} min={0} max={2} onChange={(v) => set("temperature", v)} />
        <Field label="$ / M input tokens">{(id) => <input id={id} type="number" step="0.01" className="input" value={f.price_in_per_mtok ?? ""} onChange={(e) => set("price_in_per_mtok", optNum(e.target.value))} />}</Field>
        <Field label="$ / M output tokens">{(id) => <input id={id} type="number" step="0.01" className="input" value={f.price_out_per_mtok ?? ""} onChange={(e) => set("price_out_per_mtok", optNum(e.target.value))} />}</Field>
      </div>
      <p className="muted text-xs">Prices are entered by you from the provider's price list and used only for cost estimates; leave blank if unknown.</p>
      <Toggle checked={f.enabled} onChange={(v) => set("enabled", v)} label="Enabled" />
      <ErrorBox error={save.error ?? fetchModels.error} />
      <Button variant="primary" disabled={!f.name} loading={save.isPending} onClick={() => save.mutate()}><Save className="size-4" /> {initial ? "Save provider" : "Add provider"}</Button>
    </div>
  );
}

function ProvidersTab() {
  const qc = useQueryClient();
  const toast = useToast();
  const q = useQuery({ queryKey: ["providers"], queryFn: () => api<ProvList>("/api/providers") });
  const settings = useSettings();
  const [edit, setEdit] = useState<Provider | null | "new">(null);
  const [tests, setTests] = useState<Record<number, AnyRec>>({});
  const test = useMutation({ mutationFn: (id: number) => api<AnyRec>(`/api/providers/${id}/test`, { method: "POST" }), onSuccess: (r, id) => setTests((t) => ({ ...t, [id]: r })), onError: (e: Error) => toast("error", e.message) });
  const activate = useMutation({ mutationFn: (id: number | null) => api("/api/settings/llm", { method: "PUT", body: { active_provider_id: id } }), onSuccess: () => { qc.invalidateQueries({ queryKey: ["providers"] }); qc.invalidateQueries({ queryKey: ["settings"] }); } });
  const del = useMutation({ mutationFn: (id: number) => api(`/api/providers/${id}`, { method: "DELETE" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["providers"] }) });
  const allowExternal = !!settings.data?.privacy?.allow_external_llm;
  if (!q.data) return <Spinner />;
  const active = q.data.active_provider_id;
  return (
    <div className="space-y-5">
      <Note>AI text features (inspection summaries, period summaries, Q&amp;A over analytics) are <b>optional</b>. With no active provider the app uses a deterministic, template-based summary built from the same facts. Images are never sent to any provider; only structured inspection facts are. External (cloud) providers are blocked unless allowed under <b>Privacy</b> {allowExternal ? <Badge tone="amber">currently allowed</Badge> : <Badge tone="green">currently blocked</Badge>}.</Note>
      <Card title="Providers" actions={<Button size="sm" onClick={() => setEdit("new")}><Plus className="size-4" /> Add provider</Button>} bodyClass="p-0">
        <div className="flex items-center gap-2 border-b border-slate-100 px-4 py-3 text-sm dark:border-slate-800">
          <input type="radio" name="active-prov" id="prov-none" checked={active == null} onChange={() => activate.mutate(null)} /><label htmlFor="prov-none"><b>Built-in template summaries</b> (no LLM, offline)</label>
        </div>
        {q.data.providers.filter((p) => !p.name.includes("(deleted")).map((p) => {
          const t = tests[p.id];
          return (
            <div key={p.id} className="border-b border-slate-100 px-4 py-3 text-sm dark:border-slate-800">
              <div className="flex flex-wrap items-center gap-2">
                <input type="radio" name="active-prov" id={`prov-${p.id}`} checked={active === p.id} disabled={!p.enabled} onChange={() => activate.mutate(p.id)} />
                <label htmlFor={`prov-${p.id}`} className="font-semibold">{p.name}</label>
                <Badge tone="slate">{KIND_LABEL[p.kind] ?? p.kind}</Badge>{p.external ? <Badge tone="amber">external</Badge> : <Badge tone="green">local</Badge>}{!p.enabled && <Badge tone="slate">disabled</Badge>}
                <span className="muted text-xs">{p.model || "no model set"} · {p.effective_base_url} · key {p.api_key_set ? p.api_key_masked : "not set"}</span>
                <div className="ml-auto flex gap-1">
                  <Button size="sm" onClick={() => test.mutate(p.id)} loading={test.isPending && test.variables === p.id}><Plug className="size-3.5" /> Test connection</Button>
                  <Button size="sm" variant="ghost" onClick={() => setEdit(p)}>Edit</Button>
                  <ConfirmButton size="sm" variant="danger" aria-label={`Delete provider ${p.name}`} confirmText={`Delete ${p.name}? Its stored key is erased.`} onConfirm={() => del.mutate(p.id)}><Trash2 className="size-3.5" /></ConfirmButton>
                </div>
              </div>
              {t && <p className={`mt-1 text-xs ${t.ok ? "text-emerald-600" : "text-red-600"}`}>{t.ok ? <><CheckCircle2 className="inline size-3.5" /> Connected in {t.latency_ms} ms · {t.models_found} models listed{t.completion_ok && <> · test completion "{t.reply}" ({t.tokens_in}+{t.tokens_out} tokens{t.cost_usd != null ? `, ≈ $${Number(t.cost_usd).toFixed(6)}` : ""})</>}</> : <><XCircle className="inline size-3.5" /> {t.error}</>}</p>}
            </div>
          );
        })}
      </Card>
      {q.data.usage.length > 0 && (
        <Card title="Usage & estimated cost" bodyClass="p-0">
          <table className="w-full text-sm"><thead><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">Provider</th><th className="th text-right">Calls</th><th className="th text-right">Input tokens</th><th className="th text-right">Output tokens</th><th className="th text-right">Est. cost</th></tr></thead>
            <tbody>{q.data.usage.map((u) => <tr key={u.provider} className="border-b border-slate-100 dark:border-slate-800/60"><td className="td">{u.provider}</td><td className="td text-right">{u.calls}</td><td className="td text-right">{u.tokens_in.toLocaleString()}</td><td className="td text-right">{u.tokens_out.toLocaleString()}</td><td className="td text-right">{u.cost_usd != null ? `$${u.cost_usd.toFixed(4)}` : "—"}</td></tr>)}</tbody></table>
        </Card>
      )}
      <Modal open={edit !== null} onClose={() => setEdit(null)} title={edit === "new" ? "Add AI provider" : `Edit ${(edit as Provider | null)?.name ?? ""}`} wide>
        {edit !== null && <ProviderForm key={edit === "new" ? "new" : edit.id} initial={edit === "new" ? null : edit} kinds={q.data.kinds} defaults={q.data.default_base_urls} onDone={() => setEdit(null)} />}
      </Modal>
    </div>
  );
}

// ---------------------------------------------------------------- privacy & retention
function PrivacyTab() {
  const priv = useSection("privacy");
  const ret = useSection("retention");
  const [job, setJob] = useState<string | null>(null);
  const run = useMutation({ mutationFn: (dry: boolean) => api<Job>(`/api/retention/run${qs({ dry_run: dry })}`, { method: "POST" }), onSuccess: (j) => setJob(j.id) });
  if (!priv.val || !ret.val) return <Spinner />;
  return (
    <div className="grid gap-5 xl:grid-cols-2">
      <Card title="Privacy">
        <div className="space-y-3">
          <Toggle checked={priv.val.allow_external_llm} onChange={(v) => priv.set("allow_external_llm", v)} label="Allow external (cloud) AI providers to receive structured inspection facts" />
          <Toggle checked={priv.val.include_filenames_in_llm_context} onChange={(v) => priv.set("include_filenames_in_llm_context", v)} label="Include file names / batch codes in AI context" />
          <Note>Images are never sent to AI providers, regardless of these settings. Local providers (Ollama on your own hardware) are always permitted.</Note>
        </div>
        <SaveBar dirty={priv.dirty} save={priv.save} onSave={() => priv.save.mutate(priv.val!)} />
      </Card>
      <Card title="Data retention" subtitle="0 = keep forever. The audit log is append-only and is never deleted by retention.">
        <div className="grid gap-3 md:grid-cols-3">
          <NumField label="Delete images after (days)" min={0} value={ret.val.image_days} onChange={(v) => ret.set("image_days", v)} hint="Inspection metadata is kept." />
          <NumField label="Delete inspection records after (days)" min={0} value={ret.val.inspection_days} onChange={(v) => ret.set("inspection_days", v)} />
          <NumField label="Delete finished jobs after (days)" min={0} value={ret.val.job_days} onChange={(v) => ret.set("job_days", v)} />
        </div>
        <div className="mt-3"><Toggle checked={ret.val.keep_reviewed_images} onChange={(v) => ret.set("keep_reviewed_images", v)} label="Keep images that have a human review (needed for feedback datasets)" /></div>
        <SaveBar dirty={ret.dirty} save={ret.save} onSave={() => ret.save.mutate(ret.val!)} />
        <div className="mt-4 flex gap-2 border-t border-slate-200 pt-4 dark:border-slate-800">
          <Button onClick={() => run.mutate(true)} loading={run.isPending && run.variables === true}>Dry run (count only)</Button>
          <ConfirmButton variant="danger" confirmText="Apply the saved retention policy now? Deleted images cannot be recovered." onConfirm={() => run.mutate(false)}>Apply retention now</ConfirmButton>
        </div>
        {job && <div className="mt-3"><JobStatus jobId={job} /></div>}
        <p className="muted mt-3 text-xs">Backups: <code className="kbd">python -m app.cli backup --out backup.tar.gz</code> (database + images + models), restore with <code className="kbd">python -m app.cli restore backup.tar.gz</code>.</p>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- audit & system
function AuditTab() {
  const [action, setAction] = useState("");
  const [actor, setActor] = useState("");
  const q = useQuery({ queryKey: ["audit", action, actor], queryFn: () => api<{ id: number; created_at: string; actor: string | null; action: string; entity_type: string; entity_id: string; details: AnyRec; hash: string }[]>(`/api/audit${qs({ action, actor, limit: 300 })}`) });
  const verify = useMutation({ mutationFn: () => api<{ valid: boolean; checked: number; broken_at?: number }>("/api/audit/verify") });
  return (
    <Card title="Audit log" subtitle="Append-only and hash-chained: each event includes the hash of the previous one, so edits or deletions are detectable." bodyClass="p-0"
      actions={<div className="flex items-center gap-2">
        <input className="input w-40 py-1 text-xs" placeholder="action prefix, e.g. model." value={action} onChange={(e) => setAction(e.target.value)} aria-label="Action filter" />
        <input className="input w-32 py-1 text-xs" placeholder="actor" value={actor} onChange={(e) => setActor(e.target.value)} aria-label="Actor filter" />
        <Button size="sm" onClick={() => verify.mutate()} loading={verify.isPending}><ShieldCheck className="size-3.5" /> Verify chain</Button>
      </div>}>
      {verify.data && <div className="px-4 pt-3">{verify.data.valid ? <Note tone="ok">Chain intact: {verify.data.checked} events verified.</Note> : <Note tone="error">Chain broken at event #{verify.data.broken_at} (after {verify.data.checked} events).</Note>}</div>}
      <div className="max-h-[600px] overflow-auto"><table className="w-full text-xs"><thead className="sticky top-0 bg-white dark:bg-slate-900"><tr className="border-b border-slate-200 dark:border-slate-800"><th className="th">#</th><th className="th">Time</th><th className="th">Actor</th><th className="th">Action</th><th className="th">Entity</th><th className="th">Details</th><th className="th">Hash</th></tr></thead>
        <tbody>{(q.data ?? []).map((e) => <tr key={e.id} className="border-b border-slate-100 align-top dark:border-slate-800/60"><td className="td">{e.id}</td><td className="td whitespace-nowrap">{dt(e.created_at)}</td><td className="td">{e.actor ?? "system"}</td><td className="td font-mono">{e.action}</td><td className="td">{e.entity_type} {e.entity_id}</td><td className="td max-w-md truncate font-mono" title={JSON.stringify(e.details)}>{JSON.stringify(e.details)}</td><td className="td font-mono text-slate-400">{e.hash}</td></tr>)}</tbody></table></div>
    </Card>
  );
}

function SystemTab() {
  const q = useQuery({ queryKey: ["system"], queryFn: () => api<AnyRec>("/api/system") });
  const h = useQuery({ queryKey: ["health-ready"], queryFn: () => api<AnyRec>("/api/health/ready") });
  if (!q.data) return <Spinner />;
  const d = q.data;
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card title="System"><KV items={[["Environment", d.env], ["Database", d.database], ["Worker mode", d.worker_mode], ["Device setting", d.device_setting], ["PyTorch", d.torch ? `${d.torch.version} · CUDA ${d.torch.cuda_available ? "available" : "not available (CPU)"} · ${d.torch.threads} threads` : "not installed"], ["Python", d.python], ["Platform", d.platform], ["Upload limits", `${d.limits.max_upload_mb} MB per image · ZIP ${d.limits.max_zip_mb} MB / ${d.limits.max_zip_files} files`], ["Inspections stored", d.inspections?.toLocaleString()], ["Models in memory", (d.loaded_models ?? []).join(", ") || "none"]]} /></Card>
      <Card title="Readiness"><pre className="overflow-auto rounded bg-slate-100 p-3 text-xs dark:bg-slate-800">{JSON.stringify(h.data ?? h.error?.message, null, 2)}</pre><p className="muted mt-2 text-xs">Endpoints: <code>/api/health</code> (liveness), <code>/api/health/ready</code> (DB, storage, worker), <code>/api/metrics</code> (Prometheus text format, admin).</p></Card>
    </div>
  );
}

export default function Settings() {
  const initial = (new URLSearchParams(window.location.search).get("tab") as TabId | null) ?? "policy";
  const [tab, setTab] = useState<TabId>(initial);
  const tabs: { id: TabId; label: string }[] = [
    { id: "policy", label: "Thresholds & policy" }, { id: "products", label: "Products" }, { id: "cameras", label: "Cameras" }, { id: "users", label: "Users" },
    { id: "providers", label: "AI providers" }, { id: "privacy", label: "Privacy & retention" }, { id: "audit", label: "Audit log" }, { id: "system", label: "System" },
  ];
  const change = (t: TabId) => { setTab(t); window.history.replaceState(null, "", `?tab=${t}`); };
  return (
    <>
      <PageHeader title="Settings" description="Administrator configuration. Every change is recorded in the audit log." />
      <Tabs value={tab} onChange={change} tabs={tabs} />
      {tab === "policy" && <PolicyTab />}
      {tab === "products" && <ProductsTab />}
      {tab === "cameras" && <CamerasTab />}
      {tab === "users" && <UsersTab />}
      {tab === "providers" && <ProvidersTab />}
      {tab === "privacy" && <PrivacyTab />}
      {tab === "audit" && <AuditTab />}
      {tab === "system" && <SystemTab />}
    </>
  );
}
