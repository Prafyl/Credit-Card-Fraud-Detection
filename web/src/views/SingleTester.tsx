import { useState } from "react";
import { api, Features, ModelInfo, Prediction, ScoreOptions } from "../api";
import { Button, Card, ErrorNote, SectionTitle, Spinner } from "../components/ui";
import { Verdict } from "../components/Verdict";

function emptyFeatures(names: string[]): Features {
  return Object.fromEntries(names.map((n) => [n, 0]));
}

export default function SingleTester({
  info,
  opts,
}: {
  info: ModelInfo;
  opts: ScoreOptions;
}) {
  const [features, setFeatures] = useState<Features>(emptyFeatures(info.features));
  const [trueLabel, setTrueLabel] = useState<number | undefined>();
  const [mode, setMode] = useState<"form" | "json">("form");
  const [jsonText, setJsonText] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadSample(kind: "fraud" | "legit") {
    setError(null);
    try {
      const [sample] = await api.samples(kind, 1);
      setFeatures(sample.features);
      setTrueLabel(sample.true_label);
      setJsonText(JSON.stringify(sample.features, null, 2));
      setResult(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  function clear() {
    setFeatures(emptyFeatures(info.features));
    setTrueLabel(undefined);
    setJsonText("");
    setResult(null);
    setError(null);
  }

  async function score() {
    setLoading(true);
    setError(null);
    try {
      let payload = features;
      if (mode === "json") {
        payload = JSON.parse(jsonText);
        setFeatures(payload);
      }
      const missing = info.features.filter((f) => !(f in payload));
      if (missing.length) throw new Error(`Missing features: ${missing.slice(0, 4).join(", ")}…`);
      setResult(await api.predict(payload, opts));
    } catch (e) {
      setError(e instanceof SyntaxError ? "That is not valid JSON." : (e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
      <Card className="p-6">
        <SectionTitle
          title="Score a single transaction"
          subtitle="Load a real held-out transaction, tweak any value, then score it."
          right={
            <div className="flex rounded-lg border border-line p-0.5 text-sm">
              {(["form", "json"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-3 py-1 capitalize ${
                    mode === m ? "bg-brand/15 text-ink" : "text-ink-muted"
                  }`}
                >
                  {m === "form" ? "Form" : "Paste JSON"}
                </button>
              ))}
            </div>
          }
        />

        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="danger" onClick={() => loadSample("fraud")}>
            Load a known fraud
          </Button>
          <Button variant="primary" onClick={() => loadSample("legit")}>
            Load a known legit
          </Button>
          <Button variant="ghost" onClick={clear}>
            Clear
          </Button>
        </div>

        {mode === "form" ? (
          <div className="mt-5 grid max-h-[420px] grid-cols-2 gap-2 overflow-y-auto pr-1 sm:grid-cols-3">
            {info.features.map((name) => (
              <label key={name} className="block">
                <span className="text-xs text-ink-faint">{name}</span>
                <input
                  type="number"
                  step="any"
                  value={features[name] ?? 0}
                  onChange={(e) =>
                    setFeatures({ ...features, [name]: parseFloat(e.target.value) || 0 })
                  }
                  className="mt-0.5 w-full rounded-lg border border-line bg-surface-inset px-2.5 py-1.5 font-mono text-sm text-ink outline-none focus:border-brand"
                />
              </label>
            ))}
          </div>
        ) : (
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            placeholder={`{\n  "V1": -1.36,\n  "V2": -0.07,\n  ...,\n  "Amount": 149.62\n}`}
            className="mt-5 h-[420px] w-full rounded-xl border border-line bg-surface-inset p-4 font-mono text-sm text-ink outline-none focus:border-brand"
          />
        )}

        <div className="mt-5 flex items-center gap-4">
          <Button onClick={score} disabled={loading}>
            {loading ? "Scoring…" : "Score transaction"}
          </Button>
          {loading && <Spinner />}
        </div>
      </Card>

      <div className="space-y-4">
        {error && <ErrorNote message={error} />}
        {result ? (
          <Verdict result={result} trueLabel={trueLabel} />
        ) : (
          <Card className="grid h-full min-h-[220px] place-items-center p-6 text-center">
            <div className="text-sm text-ink-muted">
              Load a sample or enter values, then score to see the verdict here.
            </div>
          </Card>
        )}
        {result && (
          <Card className="p-5 text-sm text-ink-muted">
            <div className="font-medium text-ink">How to read this</div>
            <p className="mt-2">
              Scored with <span className="text-ink">{result.model ?? info.model_name}</span>.
              The bar shows the model's fraud probability; the vertical marker is the active
              decision threshold ({result.threshold.toFixed(3)}), anything past it is flagged.
              Change the model or threshold in the controls above, or explore the trade-off on
              the Bank dashboard.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}
