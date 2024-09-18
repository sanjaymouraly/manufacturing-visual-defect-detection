import { CheckCircle2, AlertTriangle, X } from "lucide-react";
import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

type Toast = { id: number; kind: "ok" | "error"; text: string };
const Ctx = createContext<(kind: Toast["kind"], text: string) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((kind: Toast["kind"], text: string) => {
    const id = Date.now() + Math.random();
    setItems((x) => [...x, { id, kind, text }]);
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), kind === "error" ? 8000 : 4000);
  }, []);
  return (
    <Ctx.Provider value={push}>
      {children}
      <div className="fixed right-4 bottom-4 z-[100] flex w-96 max-w-[calc(100vw-2rem)] flex-col gap-2" aria-live="polite" role="status">
        {items.map((t) => (
          <div key={t.id} className={`card flex items-start gap-2 p-3 text-sm shadow-lg ${t.kind === "error" ? "border-red-300 dark:border-red-800" : "border-emerald-300 dark:border-emerald-800"}`}>
            {t.kind === "error" ? <AlertTriangle className="mt-0.5 size-4 shrink-0 text-red-600" /> : <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600" />}
            <span className="flex-1 break-words">{t.text}</span>
            <button aria-label="Dismiss" onClick={() => setItems((x) => x.filter((y) => y.id !== t.id))}><X className="size-4 opacity-60" /></button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
export const useToast = () => useContext(Ctx);
