import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api, ApiError } from "../lib/api";
import { Button, Field, Modal } from "./ui";

/** Change the signed-in user's password. Other sessions of the user are revoked server-side. */
export function ChangePasswordDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  const reset = () => { setCurrent(""); setNext(""); setConfirm(""); setError(null); setDone(false); };
  const close = () => { reset(); onClose(); };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (next.length < 8) return setError("The new password must be at least 8 characters.");
    if (next !== confirm) return setError("The new passwords do not match.");
    setBusy(true);
    try {
      await api("/api/auth/password", { method: "POST", body: { current_password: current, new_password: next } });
      setDone(true);
      await qc.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not change the password.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={close} title="Change password">
      {done ? (
        <div className="space-y-4">
          <p className="text-sm" role="status">Password changed. Your other sessions have been signed out.</p>
          <div className="flex justify-end"><Button variant="primary" onClick={close}>Done</Button></div>
        </div>
      ) : (
        <form onSubmit={submit} className="space-y-3">
          <Field label="Current password">{(id) => <input id={id} type="password" className="input" autoComplete="current-password" required value={current} onChange={(e) => setCurrent(e.target.value)} />}</Field>
          <Field label="New password" hint="At least 8 characters; common passwords are rejected.">{(id) => <input id={id} type="password" className="input" autoComplete="new-password" required minLength={8} value={next} onChange={(e) => setNext(e.target.value)} />}</Field>
          <Field label="Confirm new password">{(id) => <input id={id} type="password" className="input" autoComplete="new-password" required value={confirm} onChange={(e) => setConfirm(e.target.value)} />}</Field>
          {error && <p className="text-sm text-red-600 dark:text-red-400" role="alert">{error}</p>}
          <div className="flex justify-end gap-2 pt-1">
            <Button type="button" onClick={close}>Cancel</Button>
            <Button type="submit" variant="primary" loading={busy}>Change password</Button>
          </div>
        </form>
      )}
    </Modal>
  );
}
