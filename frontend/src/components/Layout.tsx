import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity, KeyRound, ShieldAlert, BarChart3, Boxes, Camera, ClipboardCheck, Database, FlaskConical, History, LayoutDashboard, ListTodo, LogOut, Menu, Moon,
  ScanSearch, Settings, ShieldHalf, Sun, UploadCloud, X,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ChangePasswordDialog } from "./ChangePassword";
import { useTheme } from "../lib/theme";
import type { Role } from "../lib/types";

type Item = { to: string; label: string; icon: ReactNode; roles?: Role[]; badge?: number };

export default function Layout() {
  const { user, logout } = useAuth();
  const [pwOpen, setPwOpen] = useState(false);
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  const queue = useQuery({ queryKey: ["queue-count"], queryFn: () => api<{ total: number }>("/api/reviews/queue?limit=1"), refetchInterval: 15000 });
  const groups: { title: string; items: Item[] }[] = [
    { title: "Operate", items: [
      { to: "/", label: "Dashboard", icon: <LayoutDashboard className="size-4" /> },
      { to: "/inspect", label: "Inspect", icon: <ScanSearch className="size-4" />, roles: ["admin", "inspector"] },
      { to: "/batch", label: "Batch", icon: <UploadCloud className="size-4" />, roles: ["admin", "inspector"] },
      { to: "/camera", label: "Camera", icon: <Camera className="size-4" />, roles: ["admin", "inspector"] },
      { to: "/reviews", label: "Review queue", icon: <ClipboardCheck className="size-4" />, badge: queue.data?.total },
      { to: "/history", label: "History", icon: <History className="size-4" /> },
    ] },
    { title: "Improve", items: [
      { to: "/analytics", label: "Analytics & reports", icon: <BarChart3 className="size-4" /> },
      { to: "/models", label: "Model lab", icon: <FlaskConical className="size-4" /> },
      { to: "/robustness", label: "Robustness lab", icon: <ShieldHalf className="size-4" /> },
      { to: "/datasets", label: "Datasets & feedback", icon: <Database className="size-4" /> },
      { to: "/jobs", label: "Jobs", icon: <ListTodo className="size-4" /> },
    ] },
    { title: "Admin", items: [{ to: "/settings", label: "Settings", icon: <Settings className="size-4" />, roles: ["admin"] }] },
  ];
  const nav = (
    <nav aria-label="Main" className="flex-1 space-y-5 overflow-y-auto px-3 py-4">
      {groups.map((g) => ({ ...g, items: g.items.filter((i) => !i.roles || (user && i.roles.includes(user.role))) }))
        .filter((g) => g.items.length > 0).map((g) => (
        <div key={g.title}>
          <p className="mb-1 px-2 text-[10px] font-semibold tracking-widest text-slate-400 uppercase">{g.title}</p>
          {g.items.map((i) => (
            <NavLink key={i.to} to={i.to} end={i.to === "/"} onClick={() => setOpen(false)}
              className={({ isActive }) => clsx("flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-sm font-medium",
                isActive ? "bg-brand-700 text-white" : "text-slate-300 hover:bg-slate-800 hover:text-white")}>
              {i.icon}<span className="flex-1">{i.label}</span>
              {!!i.badge && <span className="rounded-full bg-amber-500 px-1.5 text-[11px] font-bold text-slate-950">{i.badge}</span>}
            </NavLink>
          ))}
        </div>
      ))}
    </nav>
  );
  return (
    <div className="flex min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-white focus:p-2">Skip to content</a>
      <aside className={clsx("fixed inset-y-0 left-0 z-40 flex w-60 flex-col bg-slate-900 text-white transition-transform lg:translate-x-0 dark:bg-slate-900 dark:border-r dark:border-slate-800", open ? "translate-x-0" : "-translate-x-full")}>
        <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-4">
          <div className="grid size-8 place-items-center rounded-lg bg-brand-600"><Boxes className="size-5" /></div>
          <div className="leading-tight"><p className="font-semibold">DefectLens</p><p className="text-[11px] text-slate-400">Visual defect inspection</p></div>
          <button className="ml-auto lg:hidden" onClick={() => setOpen(false)} aria-label="Close menu"><X className="size-5" /></button>
        </div>
        {nav}
        <div className="border-t border-slate-800 p-3 text-sm">
          <div className="flex items-center gap-2">
            <div className="grid size-8 place-items-center rounded-full bg-slate-700 font-semibold uppercase">{user?.username[0]}</div>
            <div className="min-w-0 flex-1 leading-tight"><p className="truncate font-medium">{user?.full_name || user?.username}</p><p className="text-[11px] text-slate-400 capitalize">{user?.role}</p></div>
            <button onClick={toggle} className="rounded p-1.5 hover:bg-slate-800" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title="Toggle theme">{theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}</button>
            <button onClick={() => setPwOpen(true)} className="rounded p-1.5 hover:bg-slate-800" aria-label="Change password" title="Change password"><KeyRound className="size-4" /></button>
            <button onClick={() => logout()} className="rounded p-1.5 hover:bg-slate-800" aria-label="Sign out" title="Sign out"><LogOut className="size-4" /></button>
          </div>
        </div>
      </aside>
      {open && <div className="fixed inset-0 z-30 bg-black/40 lg:hidden" onClick={() => setOpen(false)} />}
      <div className="flex min-w-0 flex-1 flex-col lg:pl-60">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-200 bg-white/90 px-4 py-2 backdrop-blur lg:hidden dark:border-slate-800 dark:bg-slate-950/90">
          <button onClick={() => setOpen(true)} aria-label="Open menu"><Menu className="size-5" /></button>
          <span className="font-semibold">DefectLens</span>
          <Activity className="ml-auto size-4 text-brand-600" />
        </header>
        <main id="main" key={loc.pathname} className="mx-auto w-full max-w-[1500px] flex-1 p-4 sm:p-6">
          {user?.password_change_required && (
            <div role="alert" className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200">
              <ShieldAlert className="size-5 shrink-0" />
              <span className="flex-1">This account uses a default or temporary password. Choose your own password to secure it.</span>
              <button onClick={() => setPwOpen(true)} className="rounded-md bg-amber-600 px-3 py-1.5 font-medium text-white hover:bg-amber-700">Change password</button>
            </div>
          )}
          <Outlet />
        </main>
        <ChangePasswordDialog open={pwOpen} onClose={() => setPwOpen(false)} />
      </div>
    </div>
  );
}
