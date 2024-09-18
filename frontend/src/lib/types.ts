export type Role = "admin" | "inspector" | "viewer";
export type Decision = "PASS" | "FAIL" | "REVIEW";
export type Severity = "none" | "low" | "medium" | "high" | "critical";

export interface User { id: number; username: string; full_name: string; role: Role; is_active: boolean; created_at: string | null; last_login_at: string | null; password_change_required?: boolean }

export interface Region { bbox: [number, number, number, number]; polygon?: [number, number][]; area_px: number; area_fraction: number; max_score: number; mean_score: number; approximate?: boolean }
export interface Quality { width: number; height: number; brightness: number; contrast: number; sharpness: number; saturated_fraction: number; issues: string[]; ok: boolean }
export interface Review { id: number; inspection_id: string; reviewer_id: number; reviewer: string | null; label: "defective" | "normal"; defect_type: string; severity: Severity; comment: string; regions: number[][]; agrees_with_model: boolean | null; created_at: string }

export interface Inspection {
  id: string; created_at: string; product_key: string; model_id: number | null; model_name: string | null; algorithm: string | null;
  source: string; batch_code: string; camera_id: string; station: string; external_ref: string; job_id: string | null; filename: string;
  image_sha256: string; width: number | null; height: number | null; score: number | null; fail_threshold: number | null; pass_threshold: number | null;
  seg_threshold: number | null; normalized_score: number | null; calibrated_probability: number | null; ood_score: number | null; decision: Decision;
  severity: Severity; reasons: string[]; latency_ms: number | null; status: "ok" | "error"; error: string | null; needs_review: boolean;
  review_status: "none" | "pending" | "reviewed"; final_label: string | null; duplicate_of: string | null; images_purged: boolean; n_regions: number;
  images: { display: string | null; heatmap: string | null; overlay: string | null; mask: string | null };
  regions?: Region[]; quality?: Quality | null; defect_type?: { defect_type: string; probability: number; top?: Record<string, number>; note?: string } | null;
  secondary?: { model_name?: string; score?: number; decision?: Decision; error?: string } | null;
  reviews?: Review[]; reviewed_at?: string | null; duplicate?: boolean; categories?: string[];
}

export interface Paged<T> { items: T[]; total: number; page: number; page_size: number }

export interface Thresholds { fail_threshold: number; pass_threshold: number; seg_threshold?: number | null; ood_threshold?: number | null; method?: string; map_low?: number; map_high?: number }

export interface ModelVersion {
  id: number; name: string; product_key: string; algorithm: string; version: number; status: "candidate" | "approved" | "archived" | "rejected";
  artifact_dir: string; weights_sha256: string; thresholds: Thresholds; original_thresholds: Thresholds; calibrated: boolean; calibration: Record<string, number> | null;
  validation_metrics: Record<string, any>; test_metrics: Record<string, any>; training: Record<string, any>; dataset: Record<string, any>; config: Record<string, any>;
  history: { at: string; event: string; by?: string; notes?: string; [k: string]: unknown }[]; has_classifier: boolean; classifier_metrics: Record<string, any> | null;
  notes: string; created_at: string; approved_at: string | null; active_for: string[];
  summary: { test_auroc: number | null; test_ap: number | null; test_f1: number | null; test_miss_rate: number | null; test_frr: number | null; pixel_auroc: number | null; latency_ms: number | null; val_auroc: number | null; n_test: number | null };
}

export interface Product { key: string; name: string; description: string; active_model_id: number | null; secondary_model_id: number | null; policy: Record<string, unknown>; synthetic: boolean; active_model?: { id: number; name: string; algorithm: string; version: number; thresholds: Thresholds } | null }

export interface Job { id: string; type: string; status: "queued" | "running" | "succeeded" | "failed" | "cancelled"; progress: number; message: string; params: Record<string, any>; result: Record<string, any> | null; error: string | null; attempts: number; max_attempts: number; cancel_requested: boolean; created_by_id: number | null; created_at: string; started_at: string | null; finished_at: string | null }

export interface DatasetRec { id: number; name: string; product_key: string; kind: string; version: string; sample_count: number; manifest_sha256: string; created_at: string; notes: string; license: string | null; license_url: string | null; source: string | null; source_url: string | null; citation: string | null; synthetic: boolean | null; defect_types_annotated: boolean | null; masks_available: boolean | null; split_policy: string | null; preprocessing: string | null; description: string | null; labels: Record<string, number> | null; splits: Record<string, Record<string, number>> | null; defect_types: Record<string, number> | null; valid: boolean | null; warnings: string[]; quality_report?: Record<string, any> | null }

export interface Provider { id: number; name: string; kind: string; base_url: string; effective_base_url: string; model: string; api_key_set: boolean; api_key_masked: string; api_key_env: string; timeout_s: number; max_retries: number; max_tokens: number; temperature: number; price_in_per_mtok: number | null; price_out_per_mtok: number | null; enabled: boolean; external: boolean }

export interface AiText { text: string; generated_by: string; model_generated?: boolean; provider?: string; model?: string; tokens_in?: number; tokens_out?: number; cost_usd?: number | null; notice?: string; facts_fallback?: string }
