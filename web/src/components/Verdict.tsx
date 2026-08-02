// The result of scoring one transaction: verdict label + a probability meter that shows
// where the score sits relative to the decision threshold.
import { Prediction } from "../api";

export function ProbabilityMeter({
  probability,
  threshold,
}: {
  probability: number;
  threshold: number;
}) {
  const pct = Math.min(100, Math.max(0, probability * 100));
  const thrPct = Math.min(100, Math.max(0, threshold * 100));
  const isFraud = probability >= threshold;

  return (
    <div>
      <div className="relative h-3 w-full rounded-full bg-surface-inset">
        <div
          className={`h-3 rounded-full transition-all ${isFraud ? "bg-fraud" : "bg-legit"}`}
          style={{ width: `${pct}%` }}
        />
        {/* threshold marker */}
        <div
          className="absolute top-[-4px] h-5 w-0.5 bg-ink-muted"
          style={{ left: `${thrPct}%` }}
          title={`Decision threshold ${threshold.toFixed(3)}`}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-xs text-ink-faint">
        <span>0.0</span>
        <span className="text-ink-muted">
          threshold {threshold.toFixed(3)}
        </span>
        <span>1.0</span>
      </div>
    </div>
  );
}

export function Verdict({
  result,
  trueLabel,
}: {
  result: Prediction;
  trueLabel?: number;
}) {
  const isFraud = result.is_fraud;
  const correct = trueLabel == null ? null : Number(isFraud) === trueLabel;

  return (
    <div
      className={`rounded-2xl border p-5 ${
        isFraud ? "border-fraud/40 bg-fraud-soft/30" : "border-legit/40 bg-legit-soft/30"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span
            className={`flex h-11 w-11 items-center justify-center rounded-xl text-xl ${
              isFraud ? "bg-fraud/20 text-fraud" : "bg-legit/20 text-legit"
            }`}
          >
            {isFraud ? "⚠" : "✓"}
          </span>
          <div>
            <div
              className={`text-lg font-semibold ${isFraud ? "text-fraud" : "text-legit"}`}
            >
              {isFraud ? "Fraud flagged" : "Looks legitimate"}
            </div>
            <div className="text-sm text-ink-muted">
              Fraud probability {(result.fraud_probability * 100).toFixed(2)}%
            </div>
          </div>
        </div>
        {correct != null && (
          <span
            className={`rounded-lg px-2.5 py-1 text-xs font-medium ${
              correct ? "bg-legit/15 text-legit" : "bg-warn/15 text-warn"
            }`}
          >
            {correct ? "Matches true label" : "Differs from true label"}
          </span>
        )}
      </div>
      <div className="mt-5">
        <ProbabilityMeter
          probability={result.fraud_probability}
          threshold={result.threshold}
        />
      </div>
    </div>
  );
}
