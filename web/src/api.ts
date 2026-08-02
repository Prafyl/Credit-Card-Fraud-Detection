// Thin client for the FastAPI backend. In dev, Vite proxies /api -> http://localhost:8000
// (see vite.config.ts), so these are same-origin requests.

const BASE = import.meta.env.VITE_API_URL ?? "/api";

export type Features = Record<string, number>;

export interface TestMetrics {
  PR_AUC: number;
  ROC_AUC: number;
  precision: number;
  recall: number;
  f1: number;
  TP: number;
  FP: number;
  FN: number;
  TN: number;
}

export interface CurvePoint {
  threshold: number;
  TP: number;
  FP: number;
  FN: number;
  TN: number;
}

export interface ModelInfo {
  model_name: string;
  features: string[];
  default_threshold: number;
  test_metrics: TestMetrics;
  threshold_curve: CurvePoint[];
  n_train: number;
  n_val: number;
  n_test: number;
}

export interface ModelSummary {
  model_name: string;
  default_threshold: number;
  is_default: boolean;
  test_metrics: TestMetrics;
}

export interface ModelList {
  default_model: string;
  models: ModelSummary[];
}

/** Inference-time knobs shared by every scoring call. */
export interface ScoreOptions {
  model?: string;
  threshold?: number;
}

export interface Sample {
  features: Features;
  true_label: number;
  model_probability: number;
}

export interface Prediction {
  fraud_probability: number;
  is_fraud: boolean;
  threshold: number;
  model?: string;
}

export interface CsvRow extends Prediction {
  index: number;
  Amount: number;
  true_label?: number;
}

export interface CsvResult {
  summary: {
    filename: string;
    count: number;
    flagged: number;
    threshold: number;
    model?: string;
    actual_fraud?: number;
    caught?: number;
  };
  rows: CsvRow[];
}

/** Build a query string from the shared scoring options, omitting unset values. */
function scoreQuery(opts?: ScoreOptions): string {
  const p = new URLSearchParams();
  if (opts?.model != null) p.set("model", opts.model);
  if (opts?.threshold != null) p.set("threshold", String(opts.threshold));
  const s = p.toString();
  return s ? `?${s}` : "";
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  models: () => fetch(`${BASE}/models`).then(json<ModelList>),

  modelInfo: (model?: string) =>
    fetch(`${BASE}/model-info${model != null ? `?model=${encodeURIComponent(model)}` : ""}`).then(
      json<ModelInfo>,
    ),

  samples: (kind: "fraud" | "legit" | "any", n = 1) =>
    fetch(`${BASE}/samples?kind=${kind}&n=${n}`).then(json<Sample[]>),

  predict: (features: Features, opts?: ScoreOptions) =>
    fetch(`${BASE}/predict${scoreQuery(opts)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    }).then(json<Prediction>),

  predictBatch: (transactions: Features[], opts?: ScoreOptions) =>
    fetch(`${BASE}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transactions, model: opts?.model, threshold: opts?.threshold }),
    }).then(
      json<{
        count: number;
        flagged: number;
        threshold: number;
        model?: string;
        results: Prediction[];
      }>,
    ),

  predictCsv: (file: File, opts?: ScoreOptions) => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/predict/csv${scoreQuery(opts)}`, {
      method: "POST",
      body: form,
    }).then(json<CsvResult>);
  },
};
