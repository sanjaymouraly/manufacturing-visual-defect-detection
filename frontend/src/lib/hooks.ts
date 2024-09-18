import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { DatasetRec, ModelVersion, Product } from "./types";

export const useProducts = () => useQuery({ queryKey: ["products"], queryFn: () => api<Product[]>("/api/products"), staleTime: 30_000 });
export const useModels = (product?: string) =>
  useQuery({ queryKey: ["models", product ?? ""], queryFn: () => api<ModelVersion[]>(`/api/models${product ? `?product_key=${encodeURIComponent(product)}` : ""}`) });
export const useDatasets = () => useQuery({ queryKey: ["datasets"], queryFn: () => api<DatasetRec[]>("/api/datasets") });

/** Persist a small piece of UI state (e.g. last selected product) in localStorage. */
export function remember<T>(key: string, value?: T): T | null {
  try {
    if (value !== undefined) localStorage.setItem(`dl-${key}`, JSON.stringify(value));
    const v = localStorage.getItem(`dl-${key}`);
    return v ? (JSON.parse(v) as T) : null;
  } catch {
    return null;
  }
}
