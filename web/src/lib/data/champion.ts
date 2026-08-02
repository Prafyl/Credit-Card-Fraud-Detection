// The deployed model's real numbers, transcribed from the training run.
//
// Everything here is read live from GET /model-info when the API is up. This module is the
// offline fallback, so the story still tells the truth on a laptop with no backend running
// -- which is the situation a defence demo is most likely to hit.
//
// Sources:
//   reports/model_comparison.csv  -> the eight pipelines and their test PR-AUC
//   reports/cost_analysis.csv     -> the 201-step threshold sweep below
//   src/models.py                 -> TUNED_LIGHTGBM, the champion's hyperparameters

/** Test-set totals for the champion, used to turn sweep counts into rates. */
export const TEST_SET = { frauds: 95, legit: 56_651, rows: 56_746 } as const;

export const CHAMPION = {
  name: "LightGBM + weighted",
  prAuc: 0.8276981712672012,
  rocAuc: 0.9668522597733676,
  precision: 0.9722222222222222,
  recall: 0.7368421052631579,
  f1: 0.8383233532934131,
  threshold: 0.635590465254012,
  TP: 70,
  FP: 2,
  FN: 25,
  TN: 56_649,
} as const;

/** src/models.py :: TUNED_LIGHTGBM, found by randomised search on the validation set. */
export const HYPERPARAMETERS: Array<[string, string]> = [
  ["n_estimators", "1500"],
  ["num_leaves", "15"],
  ["learning_rate", "0.10"],
  ["min_child_samples", "30"],
  ["colsample_bytree", "0.70"],
  ["reg_lambda", "0.0"],
  ["class_weight", "balanced"],
];

/** All eight pipelines, ranked. Replaced by the live list when /models answers. */
export const RANKED: Array<{ name: string; prAuc: number }> = [
  { name: "LightGBM + weighted", prAuc: 0.8277 },
  { name: "XGBoost + weighted", prAuc: 0.8157 },
  { name: "RandomForest + weighted", prAuc: 0.8133 },
  { name: "RandomForest + SMOTE", prAuc: 0.8087 },
  { name: "XGBoost + SMOTE", prAuc: 0.8075 },
  { name: "LightGBM + SMOTE", prAuc: 0.7745 },
  { name: "LogisticRegression + SMOTE", prAuc: 0.6893 },
  { name: "LogisticRegression + weighted", prAuc: 0.6879 },
];

/**
 * The champion's threshold sweep, flattened to `threshold*1000, missed frauds, false alarms`
 * triples. Stored as a string because 603 numbers written out as a literal array is 6 KB of
 * source for 1.8 KB of data.
 */
const SWEEP =
  "0,0,56651,5,20,9,10,20,7,15,22,6,20,22,6,25,22,5,30,22,5,35,22,5,40,22,5,45,22,5,50,22,5," +
  "55,22,4,60,22,4,65,22,4,70,22,4,75,22,4,80,22,4,85,22,4,90,22,4,95,22,4,100,22,4,105,22,4," +
  "110,22,4,115,22,4,120,22,4,125,22,4,130,22,4,135,23,4,140,23,4,145,23,3,150,23,3,155,23,3," +
  "160,23,3,165,23,3,170,23,3,175,23,3,180,23,3,185,23,3,190,23,3,195,23,3,200,23,3,205,23,3," +
  "210,23,3,215,23,3,220,23,3,225,24,3,230,24,3,235,24,3,240,24,3,245,24,3,250,24,3,255,24,3," +
  "260,24,3,265,24,3,270,24,3,275,24,3,280,24,3,285,24,3,290,24,3,295,24,3,300,24,3,305,24,3," +
  "310,24,3,315,25,3,320,25,3,325,25,3,330,25,3,335,25,3,340,25,3,345,25,3,350,25,3,355,25,2," +
  "360,25,2,365,25,2,370,25,2,375,25,2,380,25,2,385,25,2,390,25,2,395,25,2,400,25,2,405,25,2," +
  "410,25,2,415,25,2,420,25,2,425,25,2,430,25,2,435,25,2,440,25,2,445,25,2,450,25,2,455,25,2," +
  "460,25,2,465,25,2,470,25,2,475,25,2,480,25,2,485,25,2,490,25,2,495,25,2,500,25,2,505,25,2," +
  "510,25,2,515,25,2,520,25,2,525,25,2,530,25,2,535,25,2,540,25,2,545,25,2,550,25,2,555,25,2," +
  "560,25,2,565,25,2,570,25,2,575,25,2,580,25,2,585,25,2,590,25,2,595,25,2,600,25,2,605,25,2," +
  "610,25,2,615,25,2,620,25,2,625,25,2,630,25,2,635,25,2,640,25,2,645,25,2,650,25,2,655,25,2," +
  "660,25,2,665,25,2,670,25,2,675,25,2,680,25,2,685,25,2,690,25,2,695,25,2,700,25,2,705,25,2," +
  "710,25,2,715,25,2,720,25,2,725,25,2,730,25,2,735,25,2,740,25,2,745,25,2,750,25,2,755,25,2," +
  "760,25,2,765,25,2,770,25,2,775,25,2,780,25,2,785,25,2,790,25,2,795,25,2,800,25,2,805,25,2," +
  "810,25,2,815,25,2,820,25,2,825,25,2,830,25,2,835,25,2,840,25,2,845,25,2,850,25,2,855,25,2," +
  "860,25,2,865,25,2,870,25,2,875,26,2,880,26,2,885,26,2,890,26,2,895,26,2,900,26,2,905,26,2," +
  "910,26,2,915,26,2,920,26,2,925,26,2,930,26,2,935,26,2,940,26,2,945,26,2,950,26,2,955,26,2," +
  "960,26,2,965,26,2,970,26,2,975,26,2,980,26,2,985,26,2,990,26,2,995,26,2,1000,95,0";

export interface SweepPoint {
  threshold: number;
  TP: number;
  FP: number;
  FN: number;
  TN: number;
}

/** The sweep as confusion counts, in the same shape the API's `threshold_curve` uses. */
export const FALLBACK_CURVE: SweepPoint[] = (() => {
  const n = SWEEP.split(",").map(Number);
  const out: SweepPoint[] = [];
  for (let i = 0; i < n.length; i += 3) {
    const missed = n[i + 1];
    const alarms = n[i + 2];
    out.push({
      threshold: n[i] / 1000,
      TP: TEST_SET.frauds - missed,
      FP: alarms,
      FN: missed,
      TN: TEST_SET.legit - alarms,
    });
  }
  return out;
})();

/** Nearest sweep point to `t`. The sweep is uniform, so this is an index, not a search. */
export function pointAt(curve: SweepPoint[], t: number): SweepPoint {
  if (curve.length === 0) return { threshold: t, TP: 0, FP: 0, FN: 0, TN: 0 };
  const span = curve[curve.length - 1].threshold - curve[0].threshold || 1;
  const i = Math.round(((t - curve[0].threshold) / span) * (curve.length - 1));
  return curve[Math.min(curve.length - 1, Math.max(0, i))];
}
