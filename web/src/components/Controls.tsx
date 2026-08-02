// Global inference controls shared by every screen: which model scores, and at what
// decision threshold. Lifting these into the app shell means "Test one", "Batch" and
// "Upload CSV" all score with the same, explicit settings.
import { ModelSummary } from "../api";
import { Card } from "./ui";

export interface Controls {
  model: string;
  threshold: number;
}

export function ControlsBar({
  models,
  model,
  threshold,
  defaultThreshold,
  onModel,
  onThreshold,
}: {
  models: ModelSummary[];
  model: string;
  threshold: number;
  defaultThreshold: number;
  onModel: (m: string) => void;
  onThreshold: (t: number) => void;
}) {
  const atDefault = Math.abs(threshold - defaultThreshold) < 1e-6;
  return (
    <Card className="mt-4 p-4">
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] sm:items-end">
        {/* Model picker */}
        <label className="block">
          <span className="text-xs uppercase tracking-wide text-ink-faint">Model</span>
          <select
            value={model}
            onChange={(e) => onModel(e.target.value)}
            className="mt-1 w-full rounded-lg border border-line bg-surface-inset px-3 py-2 text-sm text-ink outline-none focus:border-brand"
          >
            {models.map((m) => (
              <option key={m.model_name} value={m.model_name}>
                {m.model_name} — PR-AUC {m.test_metrics.PR_AUC.toFixed(3)}
                {m.is_default ? "  (best)" : ""}
              </option>
            ))}
          </select>
        </label>

        {/* Threshold slider */}
        <div>
          <div className="flex items-end justify-between">
            <span className="text-xs uppercase tracking-wide text-ink-faint">
              Decision threshold
            </span>
            <span className="font-mono text-lg font-semibold text-brand">
              {threshold.toFixed(3)}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={threshold}
            onChange={(e) => onThreshold(Number(e.target.value))}
            className="mt-1 w-full"
          />
          <div className="mt-1 flex items-center justify-between text-xs text-ink-faint">
            <span>catch more fraud ◀ · ▶ fewer false alarms</span>
            <button
              onClick={() => onThreshold(defaultThreshold)}
              disabled={atDefault}
              className="text-brand hover:underline disabled:text-ink-faint disabled:no-underline"
            >
              {atDefault ? "at tuned default" : `reset to tuned (${defaultThreshold.toFixed(3)})`}
            </button>
          </div>
        </div>
      </div>
    </Card>
  );
}
