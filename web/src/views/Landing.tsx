// The landing page.
//
// The top of the page is one continuous flight through five chapters: a credit card, the
// feature space inside its chip, the boosted core, the routing gate, and the console the
// whole thing is summarised on. The viewport is split for the whole of it -- WebGL owns the
// left sixty percent, the reading rail owns the right forty -- so that no body text is ever
// set over moving geometry. Everything below the story is ordinary content, and the canvas
// stops rendering entirely once the story leaves the viewport.
import { motion, useReducedMotion } from "framer-motion";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api, ModelInfo, ModelSummary, Prediction, Sample } from "../api";
import { HudRail } from "../components/HudRail";
import { MagneticButton } from "../components/motion/MagneticButton";
import { Reveal, RevealGroup, RevealItem } from "../components/motion/Reveal";
import { useCountUp } from "../hooks/useCountUp";
import { useInView } from "../hooks/useInView";
import { StoryState, useStoryScroll } from "../hooks/useScrollProgress";
import { CHAMPION } from "../lib/data/champion";
import { canRender3D } from "../lib/three/capabilities";
import { curve, sampleRisk } from "../lib/three/hudState";

// three.js and the four chapters are ~700KB together. Loading them lazily keeps the scoring
// tools instant for anyone who clicks straight through to them.
const CanvasRoot = lazy(() =>
  import("../three/CanvasRoot").then((m) => ({ default: m.CanvasRoot })),
);

const DATASET_ROWS = 283_726;
const FRAUD_RATE_PCT = 0.167;

/**
 * Scroll height of the story, one entry per chapter.
 *
 * The chapter windows in three/chapters.ts divide the 0..1 progress up; these only decide
 * how much wheel it takes to cross each slice, and they are roughly proportional to those
 * windows so that a chapter with a long camera move gets the scrolling to match. The verdict
 * is given the most because it is the one chapter with something to do rather than watch.
 */
const CHAPTER_HEIGHTS = ["180svh", "240svh", "225svh", "235svh", "140svh"];

const CHAPTER_HEADINGS = [
  "The card",
  "Inside the chip: the data vector space",
  "The decision core",
  "The verdict and the threshold",
  "Final analytics",
];

/* ------------------------------------------------------------------ pieces */

function Stat({
  value,
  decimals = 0,
  suffix = "",
  label,
  hint,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  label: string;
  hint?: string;
}) {
  const { ref, inView } = useInView<HTMLDivElement>();
  const animated = useCountUp(value, inView);
  return (
    <div ref={ref} className="rounded-2xl border border-line bg-surface-raised/60 p-5 backdrop-blur-md">
      <div className="font-mono text-3xl font-semibold tabular-nums text-ink">
        {animated.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        })}
        {suffix}
      </div>
      <div className="mt-1.5 text-xs uppercase tracking-wider text-ink-faint">{label}</div>
      {hint && <div className="mt-1 text-xs text-ink-muted">{hint}</div>}
    </div>
  );
}

/** Streams real held-out transactions through the live model. */
function DetectionTheater({ model }: { model?: string }) {
  const [entries, setEntries] = useState<
    Array<{ id: number; sample: Sample; result: Prediction }>
  >([]);
  const idRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const [sample] = await api.samples("any", 1);
        const result = await api.predict(sample.features, { model });
        if (!cancelled) {
          idRef.current += 1;
          setEntries((prev) => [{ id: idRef.current, sample, result }, ...prev].slice(0, 5));
        }
      } catch {
        /* demo panel only - stays quiet if the API is unreachable */
      }
      if (!cancelled) timer = setTimeout(tick, 2000);
    };
    tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [model]);

  return (
    <div className="space-y-2">
      {entries.length === 0 && (
        <div className="py-8 text-center text-sm text-ink-faint">Connecting to the model…</div>
      )}
      {entries.map((e, i) => (
        <motion.div
          key={e.id}
          layout
          initial={{ opacity: 0, x: -24, filter: "blur(4px)" }}
          animate={{ opacity: 1 - i * 0.14, x: 0, filter: "blur(0px)" }}
          transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          className={`flex items-center justify-between rounded-xl border px-4 py-3 ${
            e.result.is_fraud
              ? "border-fraud/40 bg-fraud-soft/25"
              : "border-legit/30 bg-legit-soft/15"
          }`}
        >
          <div className="flex items-center gap-3">
            <span className={`h-2 w-2 rounded-full ${e.result.is_fraud ? "bg-fraud" : "bg-legit"}`} />
            <span className="font-mono text-sm text-ink">
              €{Number(e.sample.features.Amount ?? 0).toFixed(2)}
            </span>
            <span className={`text-xs ${e.result.is_fraud ? "text-fraud" : "text-ink-faint"}`}>
              {e.result.is_fraud ? "BLOCKED" : "approved"}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-surface-inset">
              <motion.div
                className={`h-full rounded-full ${e.result.is_fraud ? "bg-fraud" : "bg-legit"}`}
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, e.result.fraud_probability * 100)}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
            <span className="w-14 text-right font-mono text-xs text-ink-muted">
              {(e.result.fraud_probability * 100).toFixed(1)}%
            </span>
          </div>
        </motion.div>
      ))}
    </div>
  );
}

function ModelRow({ model, rank }: { model: ModelSummary; rank: number }) {
  const { ref, inView } = useInView<HTMLDivElement>();
  const pct = model.test_metrics.PR_AUC * 100;
  return (
    <div ref={ref} className="flex items-center gap-4 py-2.5">
      <span className="w-6 font-mono text-xs text-ink-faint">{rank + 1}</span>
      <span className={`w-56 truncate text-sm ${rank === 0 ? "text-ink" : "text-ink-muted"}`}>
        {model.model_name}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-inset">
        <motion.div
          className={`h-full rounded-full ${rank === 0 ? "bg-brand" : "bg-brand/45"}`}
          initial={{ width: 0 }}
          animate={{ width: inView ? `${pct}%` : 0 }}
          transition={{ duration: 1.1, delay: rank * 0.07, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
      <span className="w-14 text-right font-mono text-sm tabular-nums text-ink">
        {model.test_metrics.PR_AUC.toFixed(3)}
      </span>
    </div>
  );
}

/** The story told as ordinary prose, for reduced-motion users and machines with no WebGL. */
function StaticStory({ onLaunch }: { onLaunch: () => void }) {
  const chapters = [
    {
      eyebrow: "01 — the front door",
      title: "Sentinel: Real-Time Fraud Engine",
      body: "Follow one card payment from the chip, through twenty-nine anonymised features and fifteen hundred boosted trees, to the moment it is approved or held for review.",
    },
    {
      eyebrow: "02 — the data vector space",
      title: "Twenty-nine numbers, and none of them mean anything",
      body: "V1 to V28 are principal components of the original fields, so nobody outside the issuing bank knows what any of them measure. Only Amount survives in its own units, and Time is dropped before training. V14 and V4 are the only two columns whose shuffling costs the model anything worth measuring.",
    },
    {
      eyebrow: "03 — the decision core",
      title: "Fifteen hundred small trees, voting in log-odds",
      body: "Each tree is grown to correct the errors of the trees before it and contributes a signed weight at whichever leaf the transaction lands on. Those weights are summed and squashed through a logistic function into one probability.",
    },
    {
      eyebrow: "04 — the verdict",
      title: "The model gives a probability. A human chooses the line.",
      body: `At the tuned threshold of ${CHAMPION.threshold.toFixed(3)} the model catches ${CHAMPION.TP} of ${CHAMPION.TP + CHAMPION.FN} frauds for ${CHAMPION.FP} false alarms. Move the line down to catch more fraud and annoy more customers; move it up for the reverse.`,
    },
    {
      eyebrow: "05 — final analytics",
      title: "Two mistakes, and they are not the same size",
      body: `A false alarm costs one customer one declined payment. A missed fraud costs the full amount. At the tuned line this model makes ${CHAMPION.FP} of the first and ${CHAMPION.FN} of the second on 56,746 held-out transactions, and no threshold makes both of those numbers zero.`,
    },
  ];

  return (
    <div className="mx-auto max-w-2xl px-5 pt-24">
      {chapters.map((c) => (
        <Reveal key={c.eyebrow} className="border-b border-line py-14 last:border-0">
          <div className="font-mono text-[11px] uppercase tracking-[0.26em] text-brand">{c.eyebrow}</div>
          <h2 className="mt-3 text-3xl font-semibold leading-tight text-ink">{c.title}</h2>
          <p className="mt-4 text-[15px] leading-relaxed text-ink-muted">{c.body}</p>
        </Reveal>
      ))}
      <div className="py-10">
        <MagneticButton onClick={onLaunch}>Launch detector →</MagneticButton>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------- page */

export default function Landing({
  models,
  info,
  onLaunch,
}: {
  models: ModelSummary[];
  info: ModelInfo | null;
  onLaunch: () => void;
}) {
  const reduce = useReducedMotion();
  const [use3D] = useState(() => canRender3D());
  const storyRef = useRef<HTMLDivElement>(null);
  const [story, setStory] = useState<StoryState>({ chapter: 0, active: true });

  const onStory = useCallback((s: StoryState) => setStory(s), []);
  useStoryScroll(storyRef, onStory, use3D);

  const toResults = useCallback(() => {
    document.getElementById("results")?.scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
  }, [reduce]);

  // Real predicted probabilities for the verdict chapter, with the labels they carry.
  // Sampled deliberately unevenly -- 30 legitimate to 18 fraudulent -- because the true base
  // rate of 0.167 percent would leave the flagged zone empty for minutes at a time. The
  // honest ratio is stated in the panel beside it.
  //
  // The label is what lets the scene draw its own mistakes rather than only its decisions.
  useEffect(() => {
    if (!use3D) return;
    let cancelled = false;
    Promise.all([api.samples("legit", 30), api.samples("fraud", 18)])
      .then(([legit, fraud]) => {
        if (cancelled) return;
        sampleRisk.pool = [...legit, ...fraud].map((s) => ({
          p: s.model_probability,
          fraud: s.true_label === 1,
        }));
      })
      .catch(() => {
        /* the scene has a deterministic fallback mix if the API is unreachable */
      });
    return () => {
      cancelled = true;
    };
  }, [use3D]);

  // The sweep every threshold readout on the page is computed from. champion.ts seeds it so
  // the story is truthful with no backend running; the live one replaces it when there is.
  useEffect(() => {
    if (info?.threshold_curve?.length) curve.points = info.threshold_curve;
  }, [info]);

  const sorted = [...models].sort((a, b) => b.test_metrics.PR_AUC - a.test_metrics.PR_AUC);

  return (
    <div className="relative">
      {use3D && (
        <>
          {/* The stage: top half on a phone, left sixty percent from `lg` up. Body text is
              never rendered inside this box. */}
          <motion.div
            className="pointer-events-none fixed inset-x-0 top-0 -z-10 h-[52svh] bg-surface-stage lg:inset-y-0 lg:left-0 lg:right-[40%] lg:h-auto"
            animate={{ opacity: story.active ? 1 : 0 }}
            transition={{ duration: 0.55, ease: "easeInOut" }}
          >
            <div
              className="absolute inset-0"
              style={{
                background:
                  "radial-gradient(120% 90% at 50% 40%, rgba(24,42,72,0.55), rgba(6,7,10,0) 72%)",
              }}
            />
            <Suspense fallback={null}>
              <CanvasRoot active={story.active} />
            </Suspense>
            {/* The seam. One hairline is enough to say the two columns are different
                surfaces; a border would say they are different boxes. */}
            <div
              className="absolute inset-y-0 right-0 hidden w-px lg:block"
              style={{
                background:
                  "linear-gradient(to bottom, transparent, rgba(57,135,229,0.35) 35%, rgba(57,135,229,0.35) 65%, transparent)",
              }}
            />
            <div
              className="absolute inset-x-0 bottom-0 h-px lg:hidden"
              style={{
                background:
                  "linear-gradient(to right, transparent, rgba(57,135,229,0.35), transparent)",
              }}
            />
          </motion.div>

          <HudRail
            chapter={story.active ? story.chapter : -1}
            info={info}
            models={models}
            onLaunch={onLaunch}
            onSkip={toResults}
          />
        </>
      )}

      {/* ============================================================ the story */}
      {use3D ? (
        <div ref={storyRef} className="relative">
          {CHAPTER_HEIGHTS.map((height, i) => (
            <section key={CHAPTER_HEADINGS[i]} style={{ height }} aria-label={CHAPTER_HEADINGS[i]}>
              {/* The chapter's copy lives in the fixed rail, so all this section carries is
                  scroll distance and a landmark to jump to. */}
              <h2 className="sr-only">{CHAPTER_HEADINGS[i]}</h2>
            </section>
          ))}
        </div>
      ) : (
        <StaticStory onLaunch={onLaunch} />
      )}

      {/* ========================================================== the evidence */}
      <div className="relative bg-surface-inset">
        <div className="mx-auto max-w-6xl px-5">
          <section id="results" className="py-28">
            <Reveal>
              <h2 className="text-center text-3xl font-semibold text-ink">Measured, not asserted</h2>
              <p className="mx-auto mt-3 max-w-xl text-center text-ink-muted">
                Every number below comes from a held-out test set the model never saw during
                training or threshold tuning.
              </p>
            </Reveal>
            <RevealGroup className="mt-10 grid grid-cols-2 gap-4 lg:grid-cols-4">
              <RevealItem>
                <Stat
                  value={info?.test_metrics.PR_AUC ?? CHAMPION.prAuc}
                  decimals={3}
                  label="PR-AUC"
                  hint="ranking quality"
                />
              </RevealItem>
              <RevealItem>
                <Stat value={DATASET_ROWS} label="Transactions" hint="after de-duplication" />
              </RevealItem>
              <RevealItem>
                <Stat value={FRAUD_RATE_PCT} decimals={3} suffix="%" label="Fraud rate" hint="1 in 600" />
              </RevealItem>
              <RevealItem>
                <Stat
                  value={(info?.test_metrics.recall ?? CHAMPION.recall) * 100}
                  decimals={1}
                  suffix="%"
                  label="Recall"
                  hint="frauds caught"
                />
              </RevealItem>
            </RevealGroup>
          </section>

          <section className="py-16">
            <Reveal className="rounded-2xl border border-line bg-surface-raised/60 p-7 backdrop-blur-md">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-ink">Live scoring</h2>
                  <p className="mt-1 text-sm text-ink-muted">
                    Real held-out transactions, scored by the running model.
                  </p>
                </div>
                <span className="flex items-center gap-2 rounded-full border border-legit/30 bg-legit-soft/20 px-3 py-1 text-xs text-legit">
                  <span className="h-1.5 w-1.5 animate-glow-pulse rounded-full bg-legit" />
                  live
                </span>
              </div>
              <div className="mt-6">
                <DetectionTheater model={info?.model_name} />
              </div>
            </Reveal>
          </section>

          <section className="py-16">
            <Reveal className="rounded-2xl border border-line bg-surface-raised/60 p-7 backdrop-blur-md">
              <h2 className="text-xl font-semibold text-ink">Eight pipelines, ranked</h2>
              <p className="mt-1 text-sm text-ink-muted">
                Four algorithms × two imbalance strategies, ranked by PR-AUC on the test set.
                Under five-fold cross-validation the top four are statistically
                indistinguishable.
              </p>
              <div className="mt-6 divide-y divide-line">
                {sorted.map((m, i) => (
                  <ModelRow key={m.model_name} model={m} rank={i} />
                ))}
              </div>
            </Reveal>
          </section>

          <section className="py-28 text-center">
            <Reveal>
              <h2 className="text-4xl font-semibold text-ink">Score a transaction</h2>
              <p className="mx-auto mt-3 max-w-md text-ink-muted">
                Load a real held-out case, move the threshold, and watch the verdict change.
              </p>
              <div className="mt-9">
                <MagneticButton onClick={onLaunch}>Launch detector →</MagneticButton>
              </div>
            </Reveal>
          </section>
        </div>
      </div>
    </div>
  );
}
