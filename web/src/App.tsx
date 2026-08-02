import { useEffect, useState } from "react";
import { AnimatePresence, motion, MotionConfig } from "framer-motion";
import { api, ModelInfo, ModelSummary } from "./api";
import { Skeleton, ErrorNote } from "./components/ui";
import { ControlsBar } from "./components/Controls";
import SingleTester from "./views/SingleTester";
import BatchTester from "./views/BatchTester";
import CsvTester from "./views/CsvTester";
import Dashboard from "./views/Dashboard";
import Landing from "./views/Landing";

type TabId = "single" | "batch" | "csv" | "dashboard";
type View = "landing" | "app";

const TABS: { id: TabId; label: string; blurb: string }[] = [
  { id: "single", label: "Test one", blurb: "Score a single transaction" },
  { id: "batch", label: "Batch", blurb: "Score several at once" },
  { id: "csv", label: "Upload CSV", blurb: "Score a whole file" },
  { id: "dashboard", label: "Bank dashboard", blurb: "Model performance & threshold" },
];

export default function App() {
  const [view, setView] = useState<View>("landing");
  const [tab, setTab] = useState<TabId>("single");
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [model, setModel] = useState<string | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load the model list once, then select the default.
  useEffect(() => {
    api
      .models()
      .then((list) => {
        setModels(list.models);
        setModel(list.default_model);
      })
      .catch((e) => setError(e.message));
  }, []);

  // Whenever the selected model changes, pull its info and reset the threshold to that
  // model's own tuned default (each model has a different sensible operating point).
  useEffect(() => {
    if (!model) return;
    setInfo(null);
    api
      .modelInfo(model)
      .then((i) => {
        setInfo(i);
        setThreshold(i.default_threshold);
      })
      .catch((e) => setError(e.message));
  }, [model]);

  const ready = info && model && threshold != null;
  const opts = ready ? { model: model!, threshold: threshold! } : undefined;

  if (view === "landing") {
    return (
      // reducedMotion="user" makes every framer-motion animation in the tree respect the OS
      // setting, so the accessibility behaviour is declared once rather than per component.
      <MotionConfig reducedMotion="user">
        <AnimatePresence mode="wait">
          <motion.div
            key="landing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
            className="min-h-full"
          >
            <Landing models={models} info={info} onLaunch={() => setView("app")} />
          </motion.div>
        </AnimatePresence>
      </MotionConfig>
    );
  }

  return (
    <MotionConfig reducedMotion="user">
    <motion.div
      key="app"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto flex min-h-full max-w-6xl flex-col px-5 pb-16"
    >
      {/* Header */}
      <header className="sticky top-0 z-10 -mx-5 flex flex-col gap-4 border-b border-line/60 bg-surface-inset/80 px-5 py-5 backdrop-blur sm:flex-row sm:items-center sm:justify-between">
        <button
          onClick={() => setView("landing")}
          className="flex items-center gap-3 text-left transition hover:opacity-80"
        >
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand/15 text-2xl">
            🛡️
          </div>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Sentinel</h1>
            <p className="text-sm text-ink-muted">Credit card fraud detection</p>
          </div>
        </button>
        {info && (
          <div className="flex items-center gap-2 text-sm">
            <span className="rounded-lg border border-line bg-surface-raised px-3 py-1.5 text-ink-muted">
              Model <span className="text-ink">{info.model_name}</span>
            </span>
            <span className="rounded-lg border border-brand/30 bg-brand/10 px-3 py-1.5 text-brand">
              PR-AUC {info.test_metrics.PR_AUC.toFixed(3)}
            </span>
          </div>
        )}
      </header>

      {/* Tab bar */}
      <nav className="mt-6 flex flex-wrap gap-1 rounded-2xl border border-line bg-surface-raised p-1.5">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex-1 rounded-xl px-4 py-2.5 text-left transition-all duration-300 sm:text-center ${
              tab === t.id
                ? "bg-brand/15 text-ink shadow-glow-brand"
                : "text-ink-muted hover:bg-white/5"
            }`}
          >
            <div className="text-sm font-medium">{t.label}</div>
            <div className="hidden text-xs text-ink-faint sm:block">{t.blurb}</div>
          </button>
        ))}
      </nav>

      {/* Global inference controls: pick the model and the decision threshold */}
      {ready && (
        <ControlsBar
          models={models}
          model={model!}
          threshold={threshold!}
          defaultThreshold={info!.default_threshold}
          onModel={setModel}
          onThreshold={setThreshold}
        />
      )}

      {/* Body */}
      <main className="mt-6 flex-1">
        {error && (
          <ErrorNote
            message={`Cannot reach the API (${error}). Start it with:  uvicorn api.main:app --port 8000`}
          />
        )}
        {!info && !error && (
          <div className="space-y-3 py-6">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        )}
        {ready && (
          <AnimatePresence mode="wait">
            <motion.div
              key={tab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
            >
              {tab === "single" && <SingleTester info={info!} opts={opts!} />}
              {tab === "batch" && <BatchTester opts={opts!} />}
              {tab === "csv" && <CsvTester info={info!} opts={opts!} />}
              {tab === "dashboard" && (
                <Dashboard
                  info={info!}
                  models={models}
                  threshold={threshold!}
                  onThreshold={setThreshold}
                />
              )}
            </motion.div>
          </AnimatePresence>
        )}
      </main>

      <footer className="mt-12 border-t border-line pt-6 text-center text-xs text-ink-faint">
        Trained on the ULB credit-card fraud dataset · features V1–V28 are anonymised PCA
        components, so use the sample loaders to populate the form.
      </footer>
    </motion.div>
    </MotionConfig>
  );
}
