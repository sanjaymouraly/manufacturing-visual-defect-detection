import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ChevronLeft, ChevronRight, Info, Loader2, X } from "lucide-react";
import { useEffect, useId, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import type { Decision, Severity } from "../lib/types";

type BtnVariant = "primary" | "secondary" | "danger" | "ghost" | "success" | "warning";
export function Button({ variant = "secondary", size = "md", loading, className, children, disabled, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: BtnVariant; size?: "sm" | "md"; loading?: boolean }) {
  const v: Record<BtnVariant, string> = {
    primary: "bg-brand-700 text-white hover:bg-brand-800 dark:bg-brand-600 dark:hover:bg-brand-700",
    secondary: "border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800",
    danger: "bg-red-600 text-white hover:bg-red-700",
    success: "bg-emerald-600 text-white hover:bg-emerald-700",
    warning: "bg-amber-500 text-slate-950 hover:bg-amber-600",
    ghost: "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800",
  };
  return (
    <button {...rest} disabled={disabled || loading}
      className={clsx("inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-2 text-sm", v[variant], className)}>
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function Card({ title, actions, children, className, bodyClass, subtitle }: { title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; bodyClass?: string }) {
  return (
    <section className={clsx("card", className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <div>
            {title && <h2 className="text-sm font-semibold">{title}</h2>}
            {subtitle && <p className="muted text-xs">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx(bodyClass ?? "p-4")}>{children}</div>
    </section>
  );
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description && <p className="muted mt-0.5 max-w-3xl text-sm">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

const decisionStyle: Record<string, string> = {
  PASS: "bg-emerald-100 text-emerald-800 ring-emerald-600/20 dark:bg-emerald-950 dark:text-emerald-300",
  FAIL: "bg-red-100 text-red-800 ring-red-600/20 dark:bg-red-950 dark:text-red-300",
  REVIEW: "bg-amber-100 text-amber-900 ring-amber-600/20 dark:bg-amber-950 dark:text-amber-300",
};
export function DecisionBadge({ decision, large }: { decision: Decision | string; large?: boolean }) {
  return <span className={clsx("inline-flex items-center rounded-md font-bold tracking-wide ring-1 ring-inset", large ? "px-3 py-1 text-lg" : "px-2 py-0.5 text-xs", decisionStyle[decision] ?? "bg-slate-100")}>{decision}</span>;
}

const sevStyle: Record<string, string> = {
  none: "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
  low: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  medium: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  high: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300",
  critical: "bg-red-600 text-white",
};
export function SeverityBadge({ severity }: { severity: Severity | string }) {
  return <span className={clsx("inline-flex rounded px-1.5 py-0.5 text-[11px] font-medium capitalize", sevStyle[severity] ?? sevStyle.none)}>{severity}</span>;
}

export function Badge({ children, tone = "slate", title }: { children: ReactNode; tone?: "slate" | "green" | "red" | "amber" | "blue" | "violet" | "teal"; title?: string }) {
  const t = {
    slate: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    green: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
    red: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
    amber: "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-300",
    blue: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
    violet: "bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-300",
    teal: "bg-brand-100 text-brand-800 dark:bg-brand-800/40 dark:text-brand-100",
  }[tone];
  return <span title={title} className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap", t)}>{children}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  const tone = ({ approved: "green", succeeded: "green", ok: "green", reviewed: "green", running: "blue", queued: "slate", candidate: "blue", pending: "amber", failed: "red", error: "red", rejected: "red", cancelled: "slate", archived: "slate" } as const)[status] ?? "slate";
  return <Badge tone={tone}>{status}</Badge>;
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return <div className="muted flex items-center gap-2 p-6 text-sm" role="status"><Loader2 className="size-4 animate-spin" />{label}…</div>;
}

export function Empty({ title, children, icon }: { title: string; children?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      {icon && <div className="text-slate-400">{icon}</div>}
      <p className="font-medium">{title}</p>
      {children && <div className="muted max-w-md text-sm">{children}</div>}
    </div>
  );
}

export function ErrorBox({ error, title = "Something went wrong" }: { error: unknown; title?: string }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div><p className="font-medium">{title}</p><p className="break-words">{msg}</p></div>
    </div>
  );
}

const noteStyle = {
  info: "border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-900 dark:bg-sky-950/40 dark:text-sky-200",
  warn: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200",
  ok: "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200",
  error: "border-red-300 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200",
};
export function Note({ children, tone = "info" }: { children: ReactNode; tone?: keyof typeof noteStyle }) {
  const Icon = tone === "warn" || tone === "error" ? AlertTriangle : tone === "ok" ? CheckCircle2 : Info;
  return (
    <div role={tone === "error" ? "alert" : undefined} className={clsx("flex items-start gap-2 rounded-lg border p-3 text-xs", noteStyle[tone])}>
      <Icon className="mt-px size-4 shrink-0" />
      <div className="space-y-1">{children}</div>
    </div>
  );
}

export function Stat({ label, value, hint, tone }: { label: string; value: ReactNode; hint?: ReactNode; tone?: "green" | "red" | "amber" }) {
  const c = tone === "green" ? "text-emerald-600 dark:text-emerald-400" : tone === "red" ? "text-red-600 dark:text-red-400" : tone === "amber" ? "text-amber-600 dark:text-amber-400" : "";
  return (
    <div className="card p-4">
      <p className="muted text-xs font-medium tracking-wide uppercase">{label}</p>
      <p className={clsx("mt-1 text-2xl font-semibold tabular-nums", c)}>{value}</p>
      {hint && <p className="muted mt-0.5 text-xs">{hint}</p>}
    </div>
  );
}

export function Field({ label, children, hint, className }: { label: string; children: (id: string) => ReactNode; hint?: ReactNode; className?: string }) {
  const id = useId();
  return (
    <div className={className}>
      <label htmlFor={id} className="label">{label}</label>
      {children(id)}
      {hint && <p className="muted mt-1 text-[11px]">{hint}</p>}
    </div>
  );
}

export function Toggle({ checked, onChange, label, disabled, description }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; description?: ReactNode }) {
  return (
    <label className={clsx("flex cursor-pointer items-start gap-3", disabled && "cursor-not-allowed opacity-60")}>
      <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={clsx("relative mt-0.5 inline-flex h-5 w-9 shrink-0 rounded-full transition-colors", checked ? "bg-brand-600" : "bg-slate-300 dark:bg-slate-700")}>
        <span className={clsx("absolute top-0.5 size-4 rounded-full bg-white shadow transition-transform", checked ? "translate-x-4.5" : "translate-x-0.5")} />
      </button>
      <span className="text-sm">
        <span className="font-medium">{label}</span>
        {description && <span className="muted block text-xs">{description}</span>}
      </span>
    </label>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: ReactNode }[]; value: T; onChange: (t: T) => void }) {
  return (
    <div role="tablist" className="mb-4 flex gap-1 overflow-x-auto border-b border-slate-200 dark:border-slate-800">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id} onClick={() => onChange(t.id)}
          className={clsx("-mb-px border-b-2 px-3 py-2 text-sm font-medium whitespace-nowrap", value === t.id ? "border-brand-600 text-brand-700 dark:text-brand-500" : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200")}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Modal({ open, onClose, title, children, wide }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    ref.current?.focus();
    return () => window.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={ref} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title} className={clsx("card max-h-[90vh] w-full overflow-y-auto outline-none", wide ? "max-w-none" : "max-w-lg")}>
        <header className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
          <h2 className="font-semibold">{title}</h2>
          <button onClick={onClose} aria-label="Close"><X className="size-5 opacity-60 hover:opacity-100" /></button>
        </header>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

export function ConfirmButton({ onConfirm, children, confirmText = "Are you sure?", ...rest }: Parameters<typeof Button>[0] & { onConfirm: () => void; confirmText?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button {...rest} onClick={() => setOpen(true)}>{children}</Button>
      <Modal open={open} onClose={() => setOpen(false)} title="Please confirm">
        <p className="text-sm">{confirmText}</p>
        <div className="mt-4 flex justify-end gap-2">
          <Button onClick={() => setOpen(false)}>Cancel</Button>
          <Button variant={rest.variant === "danger" ? "danger" : "primary"} onClick={() => { setOpen(false); onConfirm(); }}>Confirm</Button>
        </div>
      </Modal>
    </>
  );
}

export function Pagination({ page, pageSize, total, onPage }: { page: number; pageSize: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  return (
    <div className="flex items-center justify-between gap-2 px-3 py-2 text-sm">
      <span className="muted">{total.toLocaleString()} result{total === 1 ? "" : "s"} · page {page} of {pages}</span>
      <div className="flex gap-1">
        <Button size="sm" disabled={page <= 1} onClick={() => onPage(page - 1)} aria-label="Previous page"><ChevronLeft className="size-4" /></Button>
        <Button size="sm" disabled={page >= pages} onClick={() => onPage(page + 1)} aria-label="Next page"><ChevronRight className="size-4" /></Button>
      </div>
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800" role="progressbar" aria-valuenow={Math.round(value * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={label ?? "Progress"}>
        <div className="h-full rounded-full bg-brand-600 transition-all" style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
      </div>
    </div>
  );
}

export function KV({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="muted whitespace-nowrap">{k}</dt>
          <dd className="min-w-0 break-words">{v}</dd>
        </div>
      ))}
    </dl>
  );
}
