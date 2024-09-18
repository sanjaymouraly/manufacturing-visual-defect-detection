import { Boxes, Loader2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { login } = useAuth();
  const [username, setU] = useState("");
  const [password, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try { await login(username, password); } catch (x) { setErr(x instanceof Error ? x.message : "Login failed"); } finally { setBusy(false); }
  };
  return (
    <div className="grid min-h-screen place-items-center bg-gradient-to-br from-slate-900 via-slate-900 to-brand-800 p-4">
      <form onSubmit={submit} className="card w-full max-w-sm p-6" aria-label="Sign in">
        <div className="mb-6 flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-brand-700 text-white"><Boxes className="size-6" /></div>
          <div><h1 className="text-lg font-semibold">DefectLens</h1><p className="muted text-xs">Manufacturing visual defect inspection</p></div>
        </div>
        <label className="label" htmlFor="u">Username</label>
        <input id="u" className="input mb-3" autoComplete="username" value={username} onChange={(e) => setU(e.target.value)} required autoFocus />
        <label className="label" htmlFor="p">Password</label>
        <input id="p" type="password" className="input mb-4" autoComplete="current-password" value={password} onChange={(e) => setP(e.target.value)} required />
        {err && <p role="alert" className="mb-3 rounded-lg bg-red-50 p-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{err}</p>}
        <button className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand-700 py-2 text-sm font-semibold text-white hover:bg-brand-800 disabled:opacity-60" disabled={busy}>
          {busy && <Loader2 className="size-4 animate-spin" />}Sign in
        </button>
        <p className="muted mt-4 text-[11px] leading-relaxed">Decision-support system: automated results must be verified according to your quality procedure. Contact your administrator for an account.</p>
      </form>
    </div>
  );
}
