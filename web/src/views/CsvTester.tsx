import { useRef, useState } from "react";
import { api, CsvResult, ModelInfo, ScoreOptions } from "../api";
import { Button, Card, ErrorNote, SectionTitle, Spinner, Stat } from "../components/ui";

// Turn pasted text into a CSV File the /predict/csv endpoint can read. Handles three things
// a person is likely to paste: rows copied with the header line, raw data rows with no header,
// and rows copied out of a spreadsheet (tab separated). When there is no header we add one by
// matching the column count to the known layout of the dataset (Time, V1..V28, Amount, Class).
function textToCsvFile(raw: string, featureNames: string[]): File {
  let lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (!lines.length) throw new Error("No rows found. Paste some CSV rows first.");

  // Spreadsheet copy uses tabs; convert to commas when there are no commas already.
  lines = lines.map((l) => (l.includes("\t") && !l.includes(",") ? l.split("\t").join(",") : l));

  const firstFields = lines[0].split(",").map((s) => s.trim());
  const numeric = (v: string) => v !== "" && Number.isFinite(Number(v));
  const hasHeader = !firstFields.every(numeric);

  if (!hasHeader) {
    const n = firstFields.length;
    const f = featureNames;
    let header: string[] | null = null;
    if (n === f.length) header = f; // V1..V28, Amount
    else if (n === f.length + 1) header = ["Time", ...f]; // Time + features
    else if (n === f.length + 2) header = ["Time", ...f, "Class"]; // full raw row
    if (!header)
      throw new Error(
        `Each row has ${n} values, which does not match the expected ${f.length} features ` +
          `(V1..V28, Amount). Add a header row naming the columns, then try again.`,
      );
    lines = [header.join(","), ...lines];
  }

  return new File([lines.join("\n")], "pasted.csv", { type: "text/csv" });
}

export default function CsvTester({
  info,
  opts,
}: {
  info: ModelInfo;
  opts: ScoreOptions;
}) {
  const [mode, setMode] = useState<"file" | "paste">("file");
  const [pasteText, setPasteText] = useState("");
  const [result, setResult] = useState<CsvResult | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function runScore(file: File, label: string) {
    setFileName(label);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.predictCsv(file, opts));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file: File) {
    runScore(file, file.name);
  }

  function scorePaste() {
    setError(null);
    try {
      const file = textToCsvFile(pasteText, info.features);
      runScore(file, "pasted rows");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  // Pull a few held-out transactions and drop them into the box as headed CSV, with no Class
  // column, so it is easy to see the paste flow working without labels.
  async function insertExamples() {
    setError(null);
    try {
      const [fraud, legit] = await Promise.all([
        api.samples("fraud", 3),
        api.samples("legit", 3),
      ]);
      const rows = [...fraud, ...legit];
      const header = info.features.join(",");
      const body = rows
        .map((s) => info.features.map((f) => s.features[f]).join(","))
        .join("\n");
      setPasteText(`${header}\n${body}`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const s = result?.summary;
  // For a small paste, show every scored row. For a big file, keep it to the flagged ones.
  const showAll = (result?.rows.length ?? 0) <= 50;
  const displayRows = showAll ? result?.rows ?? [] : result?.rows.filter((r) => r.is_fraud) ?? [];

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <SectionTitle
          title="Score a CSV"
          subtitle="Upload a file or paste a few rows. Columns V1..V28 and Amount are used; a Class column is optional and, if present, is scored against."
          right={
            <div className="flex rounded-lg border border-line p-0.5 text-sm">
              {(["file", "paste"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-3 py-1 ${
                    mode === m ? "bg-brand/15 text-ink" : "text-ink-muted"
                  }`}
                >
                  {m === "file" ? "Upload file" : "Paste CSV"}
                </button>
              ))}
            </div>
          }
        />

        {mode === "file" ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
            }}
            className={`mt-5 grid place-items-center rounded-2xl border-2 border-dashed p-10 text-center transition ${
              dragOver ? "border-brand bg-brand/5" : "border-line"
            }`}
          >
            <div className="text-3xl">📄</div>
            <div className="mt-2 text-sm text-ink-muted">Drag a CSV here, or</div>
            <div className="mt-3">
              <Button onClick={() => inputRef.current?.click()}>Choose file</Button>
              <input
                ref={inputRef}
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />
            </div>
            {fileName && <div className="mt-3 text-xs text-ink-faint">Selected: {fileName}</div>}
          </div>
        ) : (
          <div className="mt-5">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <Button variant="subtle" onClick={insertExamples}>
                Insert 6 example rows
              </Button>
              <Button variant="ghost" onClick={() => setPasteText("")}>
                Clear
              </Button>
              <span className="text-xs text-ink-faint">
                Paste rows from your CSV. Include the header line, or paste raw rows and it will be
                added for you.
              </span>
            </div>
            <textarea
              value={pasteText}
              onChange={(e) => setPasteText(e.target.value)}
              spellCheck={false}
              placeholder={`V1,V2,...,V28,Amount\n-1.36,-0.07,...,0.13,149.62\n1.19,0.27,...,-0.02,2.69`}
              className="h-56 w-full rounded-xl border border-line bg-surface-inset p-4 font-mono text-xs text-ink outline-none focus:border-brand"
            />
            <div className="mt-3">
              <Button onClick={scorePaste} disabled={loading || !pasteText.trim()}>
                {loading ? "Scoring…" : "Score pasted rows"}
              </Button>
            </div>
          </div>
        )}

        {loading && (
          <div className="mt-4">
            <Spinner label="Scoring…" />
          </div>
        )}
        {error && (
          <div className="mt-4">
            <ErrorNote message={error} />
          </div>
        )}
      </Card>

      {s && (
        <>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Rows scored" value={s.count.toLocaleString()} />
            <Stat label="Flagged as fraud" value={String(s.flagged)} tone="bad" />
            {s.actual_fraud != null && (
              <Stat label="Actual frauds" value={String(s.actual_fraud)} tone="warn" />
            )}
            {s.caught != null && s.actual_fraud != null && (
              <Stat
                label="Caught"
                value={`${s.caught}/${s.actual_fraud}`}
                tone="good"
                hint="true frauds flagged"
              />
            )}
          </div>

          <Card className="overflow-hidden">
            <div className="flex items-center justify-between border-b border-line px-5 py-3">
              <div className="text-sm font-medium">
                {showAll ? "Scored transactions" : "Flagged transactions"}{" "}
                <span className="text-ink-faint">({displayRows.length})</span>
              </div>
              <div className="text-xs text-ink-faint">
                {s.model ?? info.model_name} · threshold {s.threshold.toFixed(3)}
              </div>
            </div>
            {displayRows.length === 0 ? (
              <div className="p-8 text-center text-sm text-ink-muted">
                Nothing was flagged in this file.
              </div>
            ) : (
              <div className="max-h-[460px] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-surface-raised text-left text-xs uppercase text-ink-faint">
                    <tr>
                      <th className="px-4 py-3 font-medium">Row</th>
                      <th className="px-4 py-3 font-medium">Amount</th>
                      <th className="px-4 py-3 font-medium">Fraud probability</th>
                      <th className="px-4 py-3 font-medium">Verdict</th>
                      <th className="px-4 py-3 font-medium">True label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayRows.map((r) => (
                      <tr key={r.index} className="border-t border-line">
                        <td className="px-4 py-2.5 text-ink-faint">{r.index}</td>
                        <td className="px-4 py-2.5 font-mono">€{r.Amount.toFixed(2)}</td>
                        <td
                          className={`px-4 py-2.5 font-mono ${
                            r.is_fraud ? "text-fraud" : "text-ink-muted"
                          }`}
                        >
                          {(r.fraud_probability * 100).toFixed(2)}%
                        </td>
                        <td className="px-4 py-2.5">
                          <span
                            className={`rounded-md px-2 py-0.5 text-xs font-medium ${
                              r.is_fraud ? "bg-fraud/15 text-fraud" : "bg-legit/15 text-legit"
                            }`}
                          >
                            {r.is_fraud ? "Fraud" : "Legit"}
                          </span>
                        </td>
                        <td className="px-4 py-2.5">
                          {r.true_label == null ? (
                            <span className="text-ink-faint">n/a</span>
                          ) : (
                            <span className={r.true_label === 1 ? "text-fraud" : "text-ink-muted"}>
                              {r.true_label === 1 ? "Fraud" : "Legit"}
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
