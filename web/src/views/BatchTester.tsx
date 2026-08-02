import { useState } from "react";
import { api, Features, Prediction, ScoreOptions } from "../api";
import { Button, Card, ErrorNote, SectionTitle, Spinner, Stat } from "../components/ui";

interface Row {
  features: Features;
  trueLabel?: number;
  result?: Prediction;
}

export default function BatchTester({ opts }: { opts: ScoreOptions }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function addSamples(kind: "fraud" | "legit", n: number) {
    setError(null);
    try {
      const samples = await api.samples(kind, n);
      setRows((prev) => [
        ...prev,
        ...samples.map((s) => ({ features: s.features, trueLabel: s.true_label })),
      ]);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function scoreAll() {
    if (!rows.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictBatch(rows.map((r) => r.features), opts);
      setRows((prev) => prev.map((r, i) => ({ ...r, result: res.results[i] })));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const scored = rows.filter((r) => r.result);
  const flagged = scored.filter((r) => r.result!.is_fraud).length;
  const caught = scored.filter((r) => r.result!.is_fraud && r.trueLabel === 1).length;
  const actualFraud = rows.filter((r) => r.trueLabel === 1).length;

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <SectionTitle
          title="Score several transactions at once"
          subtitle="Build a batch from held-out samples, then score them all in one request."
        />
        <div className="mt-4 flex flex-wrap gap-2">
          <Button variant="danger" onClick={() => addSamples("fraud", 5)}>
            + 5 frauds
          </Button>
          <Button variant="primary" onClick={() => addSamples("legit", 5)}>
            + 5 legit
          </Button>
          <Button variant="subtle" onClick={() => addSamples("legit", 20)}>
            + 20 legit
          </Button>
          <Button variant="ghost" onClick={() => setRows([])}>
            Clear
          </Button>
          <div className="ml-auto flex items-center gap-3">
            {loading && <Spinner />}
            <Button onClick={scoreAll} disabled={loading || !rows.length}>
              Score {rows.length ? `${rows.length} rows` : "batch"}
            </Button>
          </div>
        </div>
        {error && <div className="mt-4"><ErrorNote message={error} /></div>}
      </Card>

      {scored.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Scored" value={String(scored.length)} />
          <Stat label="Flagged as fraud" value={String(flagged)} tone="bad" />
          <Stat label="Actual frauds" value={String(actualFraud)} tone="warn" />
          <Stat
            label="Caught"
            value={`${caught}/${actualFraud}`}
            tone="good"
            hint="true frauds flagged"
          />
        </div>
      )}

      {rows.length > 0 && (
        <Card className="overflow-hidden">
          <div className="max-h-[460px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-raised text-left text-xs uppercase text-ink-faint">
                <tr>
                  <th className="px-4 py-3 font-medium">#</th>
                  <th className="px-4 py-3 font-medium">Amount</th>
                  <th className="px-4 py-3 font-medium">Fraud probability</th>
                  <th className="px-4 py-3 font-medium">Verdict</th>
                  <th className="px-4 py-3 font-medium">True label</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const p = r.result?.fraud_probability;
                  const fraud = r.result?.is_fraud;
                  return (
                    <tr key={i} className="border-t border-line">
                      <td className="px-4 py-2.5 text-ink-faint">{i + 1}</td>
                      <td className="px-4 py-2.5 font-mono">
                        €{(r.features.Amount ?? 0).toFixed(2)}
                      </td>
                      <td className="px-4 py-2.5 font-mono">
                        {p == null ? "—" : `${(p * 100).toFixed(2)}%`}
                      </td>
                      <td className="px-4 py-2.5">
                        {r.result == null ? (
                          <span className="text-ink-faint">not scored</span>
                        ) : (
                          <span
                            className={`rounded-md px-2 py-0.5 text-xs font-medium ${
                              fraud ? "bg-fraud/15 text-fraud" : "bg-legit/15 text-legit"
                            }`}
                          >
                            {fraud ? "Fraud" : "Legit"}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-2.5">
                        {r.trueLabel == null ? (
                          "—"
                        ) : (
                          <span
                            className={r.trueLabel === 1 ? "text-fraud" : "text-ink-muted"}
                          >
                            {r.trueLabel === 1 ? "Fraud" : "Legit"}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {rows.length === 0 && (
        <Card className="grid place-items-center p-12 text-center text-sm text-ink-muted">
          Add some samples above to build a batch.
        </Card>
      )}
    </div>
  );
}
