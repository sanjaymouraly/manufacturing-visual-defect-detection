import { useEffect } from "react";
import { remember, useProducts } from "../lib/hooks";

/** Product selector; remembers the last choice. `requireActive` hides products without an active model. */
export default function ProductSelect({ value, onChange, allowAll, requireActive, id, className }:
  { value: string; onChange: (v: string) => void; allowAll?: boolean; requireActive?: boolean; id?: string; className?: string }) {
  const { data } = useProducts();
  const list = (data ?? []).filter((p) => !requireActive || p.active_model_id);
  useEffect(() => {
    if (!value && !allowAll && list.length) {
      const last = remember<string>("product");
      onChange(list.find((p) => p.key === last)?.key ?? list[0].key);
    }
  }, [value, allowAll, list, onChange]);
  return (
    <select id={id} className={className ?? "input"} value={value} aria-label="Product"
      onChange={(e) => { onChange(e.target.value); if (e.target.value) remember("product", e.target.value); }}>
      {allowAll && <option value="">All products</option>}
      {list.map((p) => (
        <option key={p.key} value={p.key}>{p.name}{p.synthetic ? " (synthetic demo)" : ""}{!p.active_model_id ? " — no active model" : ""}</option>
      ))}
    </select>
  );
}
