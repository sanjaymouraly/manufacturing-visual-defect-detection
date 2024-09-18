import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useCallback, useContext, useEffect, type ReactNode } from "react";
import { api } from "./api";
import type { Role, User } from "./types";

interface AuthCtx { user: User | null; loading: boolean; login: (u: string, p: string) => Promise<void>; logout: () => Promise<void>; can: (...roles: Role[]) => boolean }
const Ctx = createContext<AuthCtx | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const me = useQuery({ queryKey: ["me"], queryFn: () => api<User>("/api/auth/me"), retry: false, staleTime: 60_000 });
  useEffect(() => {
    const h = () => qc.setQueryData(["me"], null);
    window.addEventListener("dl:unauthorized", h);
    return () => window.removeEventListener("dl:unauthorized", h);
  }, [qc]);
  const login = useCallback(async (username: string, password: string) => {
    const r = await api<{ user: User }>("/api/auth/login", { body: { username, password } });
    // Drop cached data of any previous session, but keep the ["me"] query itself: AuthProvider's observer
    // is subscribed to it, and qc.clear() would orphan that observer so the UI never saw the login.
    qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
    qc.setQueryData(["me"], r.user);
  }, [qc]);
  const logout = useCallback(async () => {
    try { await api("/api/auth/logout", { method: "POST" }); } finally {
      qc.removeQueries({ predicate: (q) => q.queryKey[0] !== "me" });
      qc.setQueryData(["me"], null);
    }
  }, [qc]);
  const user = me.data ?? null;
  const can = useCallback((...roles: Role[]) => !!user && roles.includes(user.role), [user]);
  return <Ctx.Provider value={{ user, loading: me.isLoading, login, logout, can }}>{children}</Ctx.Provider>;
}

export function useAuth() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useAuth outside AuthProvider");
  return c;
}
