// Turns the API's raw confusion counts into the rates the charts actually plot.
//
// The backend stores each model's threshold sweep as 201 points of {threshold, TP, FP, FN,
// TN} -- deliberately raw, so any metric can be derived client-side without re-scoring.
// Everything below is that derivation.
import { CurvePoint } from "../../api";

export interface CurveMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  tpr: number;
  fpr: number;
}

export function metricsAt(p: CurvePoint): CurveMetrics {
  const precision = p.TP + p.FP > 0 ? p.TP / (p.TP + p.FP) : 1; // no alerts => vacuously precise
  const recall = p.TP + p.FN > 0 ? p.TP / (p.TP + p.FN) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return {
    threshold: p.threshold,
    precision,
    recall,
    f1,
    tpr: recall,
    fpr: p.FP + p.TN > 0 ? p.FP / (p.FP + p.TN) : 0,
  };
}

export function curveMetrics(curve: CurvePoint[]): CurveMetrics[] {
  return curve.map(metricsAt);
}

/** The threshold with the highest F1 -- where the glowing marker sits on the 3D surface. */
export function peakF1(curve: CurveMetrics[]): CurveMetrics {
  return curve.reduce((best, p) => (p.f1 > best.f1 ? p : best), curve[0]);
}

export interface SurfaceGrid {
  models: string[];
  /** rows = models, cols = threshold steps */
  f1: number[][];
  precision: number[][];
  recall: number[][];
  thresholds: number[];
  rows: number;
  cols: number;
}

/**
 * Stacks every model's sweep into one rows x cols grid, which is exactly the vertex layout
 * a displaced PlaneGeometry wants. Models are kept in the order supplied so the surface
 * ordering matches the legend.
 */
export function buildSurfaceGrid(curves: Map<string, CurvePoint[]>, order: string[]): SurfaceGrid {
  const models = order.filter((m) => curves.has(m));
  const first = models.length ? curves.get(models[0])! : [];
  const thresholds = first.map((p) => p.threshold);

  const f1: number[][] = [];
  const precision: number[][] = [];
  const recall: number[][] = [];

  for (const model of models) {
    const m = curveMetrics(curves.get(model)!);
    f1.push(m.map((p) => p.f1));
    precision.push(m.map((p) => p.precision));
    recall.push(m.map((p) => p.recall));
  }

  return {
    models,
    f1,
    precision,
    recall,
    thresholds,
    rows: models.length,
    cols: thresholds.length,
  };
}

/** Bilinear-free lookup used when the mesh has more segments than the data has columns. */
export function sampleRow(row: number[], t: number): number {
  if (row.length === 0) return 0;
  const x = Math.min(1, Math.max(0, t)) * (row.length - 1);
  const i = Math.floor(x);
  const j = Math.min(row.length - 1, i + 1);
  return row[i] + (row[j] - row[i]) * (x - i);
}
