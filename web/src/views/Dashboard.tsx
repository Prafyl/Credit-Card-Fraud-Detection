import { lazy, Suspense, useMemo } from "react";
import { CurvePoint, ModelInfo, ModelSummary } from "../api";
import { Card, SectionTitle, Stat } from "../components/ui";
import { TradeoffChart } from "../components/TradeoffChart";
import { useModelCurves } from "../hooks/useModelCurves";

// Both 3D panels pull in three.js, so they load only once the dashboard is opened.
const ThresholdSurface3D = lazy(() =>
  import("../components/ThresholdSurface3D").then((m) => ({ default: m.ThresholdSurface3D })),
);
const MetricsBarField3D = lazy(() =>
  import("../components/MetricsBarField3D").then((m) => ({ default: m.MetricsBarField3D })),
);

function PanelFallback({ label }: { label: string }) {
  return (
    <div className="grid h-[420px] place-items-center rounded-2xl border border-line bg-surface-inset text-sm text-ink-faint">
      {label}
    </div>
  );
}

function nearest(curve: CurvePoint[], threshold: number): number {
  let best = 0;
  let bestDist = Infinity;
  curve.forEach((p, i) => {
    const d = Math.abs(p.threshold - threshold);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  });
  return best;
}

function ConfusionCell({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: number;
  sub: string;
  tone: "good" | "bad" | "warn" | "muted";
}) {
  const styles = {
    good: "border-legit/40 bg-legit-soft/30 text-legit",
    bad: "border-fraud/40 bg-fraud-soft/30 text-fraud",
    warn: "border-warn/40 bg-warn-soft/30 text-warn",
    muted: "border-line bg-surface-inset text-ink-muted",
  }[tone];
  return (
    <div className={`rounded-xl border p-4 ${styles}`}>
      <div className="text-xs uppercase tracking-wide opacity-80">{label}</div>
      <div className="mt-1 text-3xl font-semibold tabular-nums">{value.toLocaleString()}</div>
      <div className="mt-0.5 text-xs opacity-70">{sub}</div>
    </div>
  );
}

export default function Dashboard({
  info,
  models,
  threshold,
  onThreshold,
}: {
  info: ModelInfo;
  models: ModelSummary[];
  threshold: number;
  onThreshold: (t: number) => void;
}) {
  const curve = info.threshold_curve;
  // The slider is a controlled reflection of the app-wide threshold, so moving it here
  // and moving it in the top controls bar stay in lock-step.
  const idx = nearest(curve, threshold);
  const setIdx = (i: number) => onThreshold(curve[i].threshold);
  const point = curve[idx];

  const { recall, precision, missedRate } = useMemo(() => {
    const rec = point.TP + point.FN === 0 ? 0 : point.TP / (point.TP + point.FN);
    const prec = point.TP + point.FP === 0 ? 1 : point.TP / (point.TP + point.FP);
    return {
      recall: rec,
      precision: prec,
      missedRate: point.TP + point.FN === 0 ? 0 : point.FN / (point.TP + point.FN),
    };
  }, [point]);

  const m = info.test_metrics;
  const totalTx = info.n_test.toLocaleString();

  // All eight models' sweeps, for the 3D surface. Cached at module level, so revisiting the
  // dashboard does not refetch.
  const order = models.map((mm) => mm.model_name);
  const { curves } = useModelCurves(order);

  return (
    <div className="space-y-6">
      {/* Headline model quality */}
      <Card className="p-6">
        <SectionTitle
          title="Model performance"
          subtitle={`Measured on ${totalTx} held-out transactions the model never saw during training.`}
        />
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="PR-AUC"
            value={m.PR_AUC.toFixed(3)}
            hint="ranking quality (main metric)"
            tone="good"
          />
          <Stat label="ROC-AUC" value={m.ROC_AUC.toFixed(3)} hint="threshold-free" />
          <Stat
            label="Train / test"
            value={`${(info.n_train / 1000).toFixed(0)}k / ${(info.n_test / 1000).toFixed(0)}k`}
            hint="stratified split"
          />
          <Stat
            label="Fraud rate"
            value={`${(((m.TP + m.FN) / info.n_test) * 100).toFixed(2)}%`}
            hint={`${m.TP + m.FN} frauds in test`}
            tone="warn"
          />
        </div>
      </Card>

      {/* Threshold control + live confusion matrix */}
      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card className="p-6">
          <SectionTitle
            title="Operating point"
            subtitle="Move the threshold to trade off catching fraud against false alarms."
          />

          <div className="mt-6">
            <div className="flex items-end justify-between">
              <span className="text-sm text-ink-muted">Decision threshold</span>
              <span className="font-mono text-2xl font-semibold text-brand">
                {point.threshold.toFixed(3)}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={curve.length - 1}
              value={idx}
              onChange={(e) => setIdx(Number(e.target.value))}
              className="mt-3 w-full"
            />
            <div className="mt-2 flex justify-between text-xs text-ink-faint">
              <span>catch more fraud ◀ more false alarms</span>
              <span>fewer false alarms ▶ miss more fraud</span>
            </div>
            <button
              onClick={() => onThreshold(info.default_threshold)}
              className="mt-3 text-xs text-brand hover:underline"
            >
              Reset to tuned default ({info.default_threshold.toFixed(3)})
            </button>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-3">
            <Stat label="Recall" value={`${(recall * 100).toFixed(1)}%`} tone="good"
                  hint="of frauds caught" />
            <Stat label="Precision" value={`${(precision * 100).toFixed(1)}%`}
                  hint="of flags that are real" />
            <Stat label="Missed frauds" value={String(point.FN)} tone="bad"
                  hint={`${(missedRate * 100).toFixed(1)}% of frauds`} />
            <Stat label="False alarms" value={point.FP.toLocaleString()} tone="warn"
                  hint="legit flagged" />
          </div>
        </Card>

        <Card className="p-6">
          <SectionTitle
            title="Confusion matrix"
            subtitle={`At threshold ${point.threshold.toFixed(3)}, on ${totalTx} transactions.`}
          />
          <div className="mt-6 grid grid-cols-[auto_1fr_1fr] gap-2 text-sm">
            <div />
            <div className="pb-1 text-center text-xs font-medium text-ink-faint">
              Predicted fraud
            </div>
            <div className="pb-1 text-center text-xs font-medium text-ink-faint">
              Predicted legit
            </div>

            <div className="flex items-center text-xs font-medium text-ink-faint">
              <span className="-rotate-0">Actual fraud</span>
            </div>
            <ConfusionCell label="Caught" value={point.TP} sub="true positive" tone="good" />
            <ConfusionCell label="Missed" value={point.FN} sub="false negative" tone="bad" />

            <div className="flex items-center text-xs font-medium text-ink-faint">
              Actual legit
            </div>
            <ConfusionCell label="False alarm" value={point.FP} sub="false positive"
                           tone="warn" />
            <ConfusionCell label="Correct" value={point.TN} sub="true negative" tone="muted" />
          </div>
        </Card>
      </div>

      {/* 3D decision surface — the same threshold sweep as the 2D chart below, but for all
          eight models at once. The slicing plane tracks the slider above. */}
      <Card className="p-6">
        <SectionTitle
          title="Decision surface"
          subtitle="Every model's full threshold sweep as one landscape: height is F1, colour is precision. The amber plane is your current threshold — move the slider above and it walks across the surface."
        />
        <div className="mt-5">
          <Suspense fallback={<PanelFallback label="Loading 3D surface…" />}>
            <ThresholdSurface3D
              curves={curves}
              order={order}
              threshold={point.threshold}
              model={info.model_name}
            />
          </Suspense>
        </div>
      </Card>

      {/* 3D model field */}
      <Card className="p-6">
        <SectionTitle
          title="Model field"
          subtitle="Four algorithms across, two imbalance strategies deep. Bar height is PR-AUC; hover a bar to identify it."
        />
        <div className="mt-5">
          <Suspense fallback={<PanelFallback label="Loading 3D model field…" />}>
            <MetricsBarField3D models={models} active={info.model_name} />
          </Suspense>
        </div>
      </Card>

      {/* Trade-off chart */}
      <Card className="p-6">
        <SectionTitle
          title="Precision & recall across every threshold"
          subtitle="The dashed marker is the current operating point. Lower the threshold and recall rises while precision falls — the fundamental fraud-detection trade-off."
        />
        <div className="mt-4">
          <TradeoffChart curve={curve} threshold={point.threshold} />
        </div>
      </Card>

      {/* All models, ranked — the active one is highlighted. Switch models in the
          controls bar at the top of the page. */}
      <Card className="overflow-hidden">
        <div className="border-b border-line px-6 py-4">
          <SectionTitle
            title="All models compared"
            subtitle="Ranked by PR-AUC on the held-out test set. The highlighted row is active — change it from the Model control at the top."
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-ink-faint">
              <tr>
                <th className="px-6 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">PR-AUC</th>
                <th className="px-4 py-3 font-medium">ROC-AUC</th>
                <th className="px-4 py-3 font-medium">Precision</th>
                <th className="px-4 py-3 font-medium">Recall</th>
                <th className="px-4 py-3 font-medium">F1</th>
              </tr>
            </thead>
            <tbody>
              {models.map((mm) => {
                const active = mm.model_name === info.model_name;
                const t = mm.test_metrics;
                return (
                  <tr
                    key={mm.model_name}
                    className={`border-t border-line ${active ? "bg-brand/10" : ""}`}
                  >
                    <td className="px-6 py-2.5">
                      <span className={active ? "font-medium text-ink" : "text-ink-muted"}>
                        {mm.model_name}
                      </span>
                      {mm.is_default && (
                        <span className="ml-2 rounded bg-brand/15 px-1.5 py-0.5 text-xs text-brand">
                          best
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 font-mono tabular-nums">{t.PR_AUC.toFixed(3)}</td>
                    <td className="px-4 py-2.5 font-mono tabular-nums text-ink-muted">
                      {t.ROC_AUC.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 font-mono tabular-nums text-ink-muted">
                      {t.precision.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 font-mono tabular-nums text-ink-muted">
                      {t.recall.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5 font-mono tabular-nums text-ink-muted">
                      {t.f1.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
