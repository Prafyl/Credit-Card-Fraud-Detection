// The dataset's feature vector, with the importance each column actually earned.
//
// Numbers are transcribed from reports/feature_importance.csv, produced by
// src/deep_analysis.py for the champion pipeline (LightGBM + weighted):
//   perm  = permutation importance, the drop in test PR-AUC when the column is shuffled
//   gain  = the model's own split gain, as a percentage of total gain
//
// They are hard-coded rather than fetched because the API is deliberately minimal and has
// no importance endpoint; the alternative would be adding one to the backend.

export type FeatureKind = "pca" | "amount" | "time" | "target";

export interface Feature {
  name: string;
  kind: FeatureKind;
  /** Permutation importance. Negative means shuffling the column *helped*. */
  perm: number;
  /** Share of the model's total split gain, in percent. */
  gain: number;
  note?: string;
}

const PCA: Array<[string, number, number]> = [
  ["V1", 0.002818, 0.381],
  ["V2", 0.001576, 0.744],
  ["V3", 0.011665, 1.927],
  ["V4", 0.054062, 7.56],
  ["V5", 0.004286, 0.277],
  ["V6", 0.001664, 0.414],
  ["V7", 0.01399, 0.358],
  ["V8", 0.002089, 3.15],
  ["V9", 0.001521, 0.24],
  ["V10", 0.003629, 8.62],
  ["V11", 0.010429, 0.739],
  ["V12", 0.008716, 9.653],
  ["V13", 0.005125, 0.971],
  ["V14", 0.07255, 50.912],
  ["V15", 0.003996, 0.463],
  ["V16", 0.000982, 0.748],
  ["V17", -0.004184, 4.601],
  ["V18", 0.001963, 0.715],
  ["V19", 0.004644, 1.341],
  ["V20", 0.004932, 0.753],
  ["V21", 0.004498, 0.623],
  ["V22", 0.002735, 0.735],
  ["V23", -0.001359, 0.159],
  ["V24", 0.004101, 0.351],
  ["V25", 0.00166, 0.575],
  ["V26", 0.011224, 0.948],
  ["V27", 0.006082, 0.345],
  ["V28", 0.011959, 0.733],
];

export const FEATURES: Feature[] = [
  {
    name: "Time",
    kind: "time",
    perm: 0,
    gain: 0,
    note: "seconds since the first transaction - dropped before training, it encodes only when the two-day sample was collected",
  },
  ...PCA.map(([name, perm, gain]): Feature => ({ name, kind: "pca", perm, gain })),
  {
    name: "Amount",
    kind: "amount",
    perm: 0.008203,
    gain: 0.966,
    note: "the only feature in its original units; fraud has a median of EUR 9.82 against EUR 22.00 for legitimate activity",
  },
  {
    name: "Class",
    kind: "target",
    perm: 0,
    gain: 0,
    note: "the label being predicted, never an input - 473 frauds in 283,726 rows",
  },
];

/** The 29 columns the model is actually given. */
export const MODEL_FEATURES = FEATURES.filter((f) => f.kind === "pca" || f.kind === "amount");

export const MAX_PERM = Math.max(...MODEL_FEATURES.map((f) => f.perm));

/** 0..1 importance used for node size and glow. Negative importances floor at zero. */
export function importance(f: Feature): number {
  return Math.max(0, f.perm) / MAX_PERM;
}

export const TOP_FEATURES = [...MODEL_FEATURES].sort((a, b) => b.perm - a.perm);

/** The toggle groups exposed in the HUD; the bit index matches hudState.groups. */
export const GROUPS = [
  { id: "core", label: "Top drivers", test: (f: Feature) => importance(f) > 0.12 },
  { id: "pca", label: "PCA V1-V28", test: (f: Feature) => f.kind === "pca" },
  { id: "raw", label: "Time / Amount", test: (f: Feature) => f.kind === "time" || f.kind === "amount" },
] as const;
