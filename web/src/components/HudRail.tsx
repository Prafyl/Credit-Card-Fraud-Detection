// The reading rail: the right forty percent of the viewport, and the only place on the
// landing page where body copy is allowed to live.
//
// Everything here is DOM, not 3D text. Four reasons: it stays crisp at any resolution,
// bloom cannot bleach it, the interactive parts are real form controls with real keyboard
// and screen-reader behaviour, and -- the one that decided the layout -- it can be given a
// column of its own that the WebGL canvas is not allowed to enter.
//
// Panels write into the `hud` module object rather than passing props into the canvas.
// Dragging the slider fires sixty times a second, and each of those would otherwise
// re-render the entire scene graph.
import { AnimatePresence, motion } from "framer-motion";
import { ReactNode, useEffect, useState } from "react";
import { ModelInfo, ModelSummary } from "../api";
import { CHAMPION, HYPERPARAMETERS, RANKED, pointAt } from "../lib/data/champion";
import { metricsAt } from "../lib/data/curveMath";
import { GROUPS, TOP_FEATURES } from "../lib/data/features";
import { alarm, curve, hud, routed } from "../lib/three/hudState";
import { CHAPTERS } from "../three/chapters";

/* ------------------------------------------------------------------ chrome */

/**
 * A dark glass panel with corner brackets.
 *
 * The brackets are what make it read as instrumentation rather than as a marketing card,
 * and the near-opaque fill behind the blur is what guarantees the contrast: a panel that
 * relies on blur alone goes unreadable the moment a bright particle drifts under it.
 */
function Glass({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`pointer-events-auto relative rounded-2xl border border-white/[0.07] bg-[#080A0F]/85 p-5 backdrop-blur-md ${className}`}
      style={{
        boxShadow:
          "inset 0 1px 0 rgba(255,255,255,0.06), 0 0 0 1px rgba(57,135,229,0.10), 0 30px 60px -32px rgba(0,0,0,0.95)",
      }}
    >
      {[
        "left-[-1px] top-[-1px] border-l-2 border-t-2 rounded-tl-2xl",
        "right-[-1px] top-[-1px] border-r-2 border-t-2 rounded-tr-2xl",
        "left-[-1px] bottom-[-1px] border-b-2 border-l-2 rounded-bl-2xl",
        "right-[-1px] bottom-[-1px] border-b-2 border-r-2 rounded-br-2xl",
      ].map((c) => (
        <span key={c} className={`pointer-events-none absolute h-4 w-4 border-brand/60 ${c}`} />
      ))}
      {children}
    </div>
  );
}

function Eyebrow({ index, children }: { index: number; children: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2.5 font-mono text-[10px] uppercase tracking-[0.24em] text-brand-bright">
      <span className="tabular-nums text-ink-faint">{String(index).padStart(2, "0")}</span>
      <span className="h-px w-4 bg-brand/50" />
      {children}
    </div>
  );
}

function Title({ children }: { children: ReactNode }) {
  return (
    <h2
      className="text-[27px] font-semibold leading-[1.14] text-ink"
      style={{ letterSpacing: "-0.028em", textWrap: "balance" }}
    >
      {children}
    </h2>
  );
}

function Body({ children }: { children: ReactNode }) {
  return <p className="mt-3 text-[14px] leading-[1.62] text-ink-muted">{children}</p>;
}

/** Footnotes are the first thing to go when the rail is a phone's bottom half. */
function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-4 hidden border-t border-white/[0.06] pt-3 font-mono text-[10.5px] leading-[1.6] text-ink-faint sm:block">
      {children}
    </p>
  );
}

function Row({ k, v, accent }: { k: string; v: string; accent?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[2px] font-mono text-[11.5px]">
      <span className="text-ink-faint">{k}</span>
      <span className={`tabular-nums ${accent ?? "text-ink"}`}>{v}</span>
    </div>
  );
}

/** A labelled bar. `of` is the value the bar is drawn as a fraction of. */
function Meter({
  label,
  value,
  of,
  display,
  tone = "brand",
  delay = 0,
}: {
  label: string;
  value: number;
  of: number;
  display: string;
  tone?: "brand" | "gold";
  delay?: number;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-[132px] shrink-0 truncate font-mono text-[10.5px] text-ink-muted">{label}</span>
      <div className="h-[5px] flex-1 overflow-hidden rounded-full bg-white/[0.06]">
        <motion.div
          className={`h-full rounded-full ${
            tone === "gold" ? "bg-gradient-to-r from-brand to-gold" : "bg-gradient-to-r from-brand/70 to-brand-bright"
          }`}
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(2, (value / of) * 100)}%` }}
          transition={{ duration: 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
      <span className="w-[52px] shrink-0 text-right font-mono text-[10.5px] tabular-nums text-ink-faint">
        {display}
      </span>
    </div>
  );
}

const enter = {
  initial: { opacity: 0, y: 18, filter: "blur(7px)" },
  animate: { opacity: 1, y: 0, filter: "blur(0px)" },
  exit: { opacity: 0, y: -14, filter: "blur(7px)" },
  transition: { duration: 0.42, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] },
};

/* ------------------------------------------------------------------ panels */

function HeroPanel({
  info,
  onLaunch,
  onSkip,
}: {
  info: ModelInfo | null;
  onLaunch: () => void;
  onSkip: () => void;
}) {
  const m = info?.test_metrics;
  const chips: Array<[string, string]> = [
    ["PR-AUC", (m?.PR_AUC ?? CHAMPION.prAuc).toFixed(3)],
    ["Recall", `${((m?.recall ?? CHAMPION.recall) * 100).toFixed(1)}%`],
    ["Precision", `${((m?.precision ?? CHAMPION.precision) * 100).toFixed(1)}%`],
  ];

  return (
    <motion.div {...enter}>
      <Glass>
        <Eyebrow index={1}>the front door</Eyebrow>
        <div
          className="font-mono text-[13px] uppercase text-brand-bright"
          style={{ letterSpacing: "0.42em" }}
        >
          Sentinel
        </div>
        <h1
          className="mt-2 text-[38px] font-semibold leading-[1.05] text-ink"
          style={{ letterSpacing: "-0.035em", textWrap: "balance" }}
        >
          Real-Time Fraud Engine
        </h1>
        <Body>
          Follow one card payment from the chip, through twenty-nine anonymised features and
          fifteen hundred boosted trees, to the moment it is approved or held for review.
        </Body>

        <div className="mt-5 grid grid-cols-3 gap-2">
          {chips.map(([label, value]) => (
            <div key={label} className="rounded-xl border border-white/[0.07] bg-white/[0.03] px-3 py-2.5">
              <div className="font-mono text-[17px] font-semibold tabular-nums text-ink">{value}</div>
              <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.16em] text-ink-faint">
                {label}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-2.5">
          <button
            onClick={onLaunch}
            className="rounded-xl bg-brand px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-bright focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright"
          >
            Launch detector
          </button>
          <button
            onClick={onSkip}
            className="rounded-xl border border-white/[0.1] px-5 py-2.5 text-sm text-ink-muted transition hover:border-white/20 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright"
          >
            Skip to results
          </button>
        </div>

        <Note>
          {(m?.TP ?? CHAMPION.TP)} frauds caught · {(m?.FP ?? CHAMPION.FP)} false alarms ·{" "}
          {(m?.FN ?? CHAMPION.FN)} missed, on 56,746 held-out transactions. Scroll to enter the
          chip.
        </Note>
      </Glass>
    </motion.div>
  );
}

function VectorPanel() {
  // Local mirror of the toggle state: the DOM needs it to render, the renderer reads the
  // module object, and the two are written together on click.
  const [groups, setGroups] = useState(() => ({ ...hud.groups }));

  const toggle = (id: string) => {
    const next = { ...groups, [id]: !groups[id] };
    setGroups(next);
    hud.groups = next;
  };

  const top = TOP_FEATURES.slice(0, 8);
  const max = top[0].perm;

  return (
    <motion.div {...enter}>
      <Glass>
        <Eyebrow index={2}>the data vector space</Eyebrow>
        <Title>Twenty-nine numbers, and none of them mean anything</Title>
        <Body>
          The public dataset is anonymised: V1 to V28 are principal components of the original
          fields, so nobody outside the issuing bank knows what any of them measure. Only Amount
          survives in its own units, and Time is dropped before training.
        </Body>

        {/* Below `lg` the rail is a phone's bottom half and the page scroll has to keep
            driving the story, so the panel must fit without a scrollbar of its own. */}
        <div className="mt-4 hidden flex-wrap gap-1.5 lg:flex">
          {GROUPS.map((g) => (
            <button
              key={g.id}
              onClick={() => toggle(g.id)}
              aria-pressed={groups[g.id]}
              className={`rounded-lg border px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright ${
                groups[g.id]
                  ? "border-brand/50 bg-brand/15 text-brand-bright"
                  : "border-white/[0.09] text-ink-faint hover:text-ink-muted"
              }`}
            >
              {g.label}
            </button>
          ))}
        </div>

        <div className="mt-4 hidden space-y-1.5 lg:block">
          {top.map((f, i) => (
            <Meter
              key={f.name}
              label={f.name}
              value={f.perm}
              of={max}
              display={f.perm.toFixed(4)}
              tone={i < 2 ? "gold" : "brand"}
              delay={i * 0.04}
            />
          ))}
        </div>

        <Note>
          Permutation importance: the drop in test PR-AUC when one column is shuffled. V14 and
          V4 glow gold in the space because they are the only two that cost the model anything
          worth measuring — V14 alone carries 50.9% of its total split gain.
        </Note>
      </Glass>
    </motion.div>
  );
}

function CorePanel({ models }: { models: ModelSummary[] }) {
  const ranked = models.length
    ? [...models]
        .sort((a, b) => b.test_metrics.PR_AUC - a.test_metrics.PR_AUC)
        .map((m) => ({ name: m.model_name, prAuc: m.test_metrics.PR_AUC }))
    : RANKED;

  return (
    <motion.div {...enter}>
      <Glass>
        <Eyebrow index={3}>the decision core</Eyebrow>
        <Title>Fifteen hundred small trees, voting in log-odds</Title>
        <Body>
          Each tree is grown to correct the errors of the trees before it and contributes a
          signed weight at whichever leaf the transaction lands on. The logistic gate turns
          that running sum into one probability.
        </Body>

        <div className="mt-3.5 hidden rounded-xl border border-white/[0.06] bg-white/[0.02] px-3.5 py-2 lg:block">
          <div className="mb-1.5 font-mono text-[9.5px] uppercase tracking-[0.18em] text-ink-faint">
            deployed champion · LightGBM
          </div>
          {HYPERPARAMETERS.map(([k, v]) => (
            <Row key={k} k={k} v={v} accent={k === "class_weight" ? "text-gold" : undefined} />
          ))}
        </div>

        <div className="mt-3.5 hidden space-y-1 lg:block">
          {ranked.map((m, i) => (
            <Meter
              key={m.name}
              label={m.name}
              value={m.prAuc}
              of={1}
              display={m.prAuc.toFixed(3)}
              tone={i === 0 ? "gold" : "brand"}
              delay={i * 0.05}
            />
          ))}
        </div>

        <Note>
          Four algorithms × two imbalance strategies, on the test set. Under five-fold
          cross-validation the top four are statistically indistinguishable.
        </Note>
      </Glass>
    </motion.div>
  );
}

/** The two ways an operating point fails, as the scene is currently experiencing them. */
function Stress({ label, value, tone }: { label: string; value: number; tone: "warn" | "fraud" }) {
  const hot = value > 0.66;
  return (
    <div className="flex items-center gap-2">
      <span
        className={`h-1.5 w-1.5 shrink-0 rounded-full ${
          hot ? (tone === "warn" ? "bg-warn animate-glow-pulse" : "bg-fraud animate-glow-pulse") : "bg-white/20"
        }`}
      />
      <span className="w-[104px] shrink-0 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-faint">
        {label}
      </span>
      <div className="h-[4px] flex-1 overflow-hidden rounded-full bg-white/[0.06]">
        <motion.div
          className={`h-full rounded-full ${tone === "warn" ? "bg-warn" : "bg-fraud"}`}
          animate={{ width: `${Math.max(1.5, value * 100)}%` }}
          transition={{ duration: 0.35 }}
        />
      </div>
    </div>
  );
}

function VerdictPanel() {
  const [threshold, setThreshold] = useState(hud.threshold);
  const [live, setLive] = useState({ legit: 0, fraud: 0, flood: 0, leak: 0 });

  // The renderer counts every packet it routes and republishes both stress values; polling
  // four times a second is enough for a readout and costs four renders instead of sixty.
  useEffect(() => {
    const id = setInterval(
      () => setLive({ ...routed, flood: alarm.flood, leak: alarm.leak }),
      250,
    );
    return () => clearInterval(id);
  }, []);

  // Nearest point on the model's real 201-step sweep of the test set. No re-scoring: the
  // API ships the confusion counts at every threshold, and champion.ts carries the same
  // sweep for when it is not running.
  const point = pointAt(curve.points, threshold);
  const m = metricsAt(point);
  const total = live.legit + live.fraud || 1;

  const readouts: Array<[string, string, string]> = [
    ["precision", `${(m.precision * 100).toFixed(1)}%`, "text-legit"],
    ["recall", `${(m.recall * 100).toFixed(1)}%`, "text-warn"],
    ["false alarms", String(point.FP), "text-fraud"],
    ["frauds missed", String(point.FN), "text-ink"],
  ];

  const tuned = Math.abs(threshold - CHAMPION.threshold) < 0.004;

  return (
    <motion.div {...enter}>
      <Glass>
        <Eyebrow index={4}>the verdict</Eyebrow>
        <Title>The model gives a probability. A human chooses the line.</Title>
        <Body>
          Drag the threshold and the gate turns, the blade sweeps, and the stream in front of
          you re-routes mid-flight. Down catches more fraud and annoys more customers; up does
          the reverse.
        </Body>

        <div className="mt-5 flex items-center gap-3.5">
          <input
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={threshold}
            aria-label="Decision threshold"
            onChange={(e) => {
              const v = Number(e.target.value);
              setThreshold(v);
              hud.threshold = v;
            }}
            onPointerDown={() => (hud.scrubbing = true)}
            onPointerUp={() => (hud.scrubbing = false)}
            onBlur={() => (hud.scrubbing = false)}
            className="flex-1"
          />
          <span className="w-[62px] text-right font-mono text-lg tabular-nums text-brand-bright">
            {threshold.toFixed(3)}
          </span>
        </div>

        <div className="mt-4 grid grid-cols-4 gap-3">
          {readouts.map(([label, value, tone]) => (
            <div key={label}>
              <div className={`font-mono text-[19px] tabular-nums ${tone}`}>{value}</div>
              <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
                {label}
              </div>
            </div>
          ))}
        </div>

        {/* The two failure modes, live off the stream. They are opposite ends of the same
            slider, and at a sane operating point both sit near zero. */}
        <div className="mt-4 hidden space-y-1.5 lg:block">
          <Stress label="vault flooding" value={live.flood} tone="fraud" />
          <Stress label="fraud slipping" value={live.leak} tone="warn" />
        </div>

        {/* Where the stream in front of you is actually going. */}
        <div className="mt-4 flex h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
          <motion.div
            className="h-full bg-legit"
            animate={{ width: `${(live.legit / total) * 100}%` }}
            transition={{ duration: 0.4 }}
          />
          <motion.div
            className="h-full bg-fraud"
            animate={{ width: `${(live.fraud / total) * 100}%` }}
            transition={{ duration: 0.4 }}
          />
        </div>
        <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-ink-faint">
          <span>{live.legit.toLocaleString()} approved</span>
          <button
            onClick={() => {
              setThreshold(CHAMPION.threshold);
              hud.threshold = CHAMPION.threshold;
            }}
            disabled={tuned}
            className="pointer-events-auto rounded px-1.5 py-0.5 uppercase tracking-[0.12em] transition enabled:hover:text-brand-bright disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright"
          >
            reset to tuned
          </button>
          <span>{live.fraud.toLocaleString()} flagged</span>
        </div>

        <Note>
          The tuned operating point is {CHAMPION.threshold.toFixed(3)}. Packets in front of you
          are sampled 30 legitimate to 18 fraudulent so both zones stay busy; the real rate is
          0.167 percent.
        </Note>
      </Glass>
    </motion.div>
  );
}

/** The 2x2, laid out the way it is written down. */
function Matrix({ TP, FP, FN, TN }: { TP: number; FP: number; FN: number; TN: number }) {
  const cells: Array<[string, number, string]> = [
    ["true negative", TN, "text-brand-bright"],
    ["false alarm", FP, "text-warn"],
    ["missed fraud", FN, "text-fraud"],
    ["caught", TP, "text-legit"],
  ];
  return (
    <div className="grid grid-cols-2 gap-1.5">
      {cells.map(([label, value, tone]) => (
        <div key={label} className="rounded-lg border border-white/[0.07] bg-white/[0.02] px-3 py-2">
          <div className={`font-mono text-[17px] font-semibold tabular-nums ${tone}`}>
            {value.toLocaleString()}
          </div>
          <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.14em] text-ink-faint">
            {label}
          </div>
        </div>
      ))}
    </div>
  );
}

function ConsolePanel({ onLaunch, onSkip }: { onLaunch: () => void; onSkip: () => void }) {
  // The console has no slider of its own -- the threshold belongs to the previous chapter --
  // so it polls for whatever was left there, and the pillars in front of it read the same.
  const [threshold, setThreshold] = useState(hud.threshold);
  const [fired, setFired] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setThreshold(hud.threshold), 250);
    return () => clearInterval(id);
  }, []);

  const point = pointAt(curve.points, threshold);
  const m = metricsAt(point);

  return (
    <motion.div {...enter}>
      <Glass>
        <Eyebrow index={5}>final analytics</Eyebrow>
        <Title>The whole engine, at the line you chose</Title>
        <Body>
          Every number here is the held-out test set at threshold {threshold.toFixed(3)} — the
          same sweep the pillars in front of you are standing at.
        </Body>

        <div className="mt-4">
          <Matrix TP={point.TP} FP={point.FP} FN={point.FN} TN={point.TN} />
        </div>

        <div className="mt-3 hidden grid-cols-3 gap-2 lg:grid">
          {(
            [
              ["precision", `${(m.precision * 100).toFixed(1)}%`],
              ["recall", `${(m.recall * 100).toFixed(1)}%`],
              ["F1", m.f1.toFixed(3)],
            ] as Array<[string, string]>
          ).map(([label, value]) => (
            <div key={label} className="rounded-lg border border-white/[0.07] bg-white/[0.03] px-3 py-2">
              <div className="font-mono text-[16px] font-semibold tabular-nums text-ink">{value}</div>
              <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.16em] text-ink-faint">
                {label}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => {
              hud.inject += 1;
              setFired((n) => n + 1);
            }}
            className="rounded-xl border border-fraud/45 bg-fraud-soft/30 px-4 py-2.5 text-sm font-medium text-fraud transition hover:border-fraud/70 hover:bg-fraud-soft/50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-fraud"
          >
            Inject anomaly test payment
          </button>
          <button
            onClick={onLaunch}
            className="rounded-xl bg-brand px-4 py-2.5 text-sm font-medium text-white transition hover:bg-brand-bright focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright"
          >
            Launch detector
          </button>
          <button
            onClick={onSkip}
            className="rounded-xl border border-white/[0.1] px-4 py-2.5 text-sm text-ink-muted transition hover:border-white/20 hover:text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-bright"
          >
            Full results
          </button>
        </div>

        <Note>
          The injected payment is synthetic and scores 0.970, so whether it lands in “caught”
          or in “missed fraud” is decided entirely by the threshold you left on the slider.
          {fired > 0 && ` Fired ${fired} time${fired === 1 ? "" : "s"}.`}
        </Note>
      </Glass>
    </motion.div>
  );
}

/* -------------------------------------------------------------- progress */

/**
 * One tick per chapter, and the name of the one you are on.
 *
 * Naming them all at once needs more width than the rail has and wraps into two ragged lines
 * the moment the window is narrower than a laptop. The position is what the strip is for;
 * only the current chapter needs a name.
 */
function Progress({ chapter }: { chapter: number }) {
  const current = CHAPTERS[chapter];
  return (
    <div
      className="pointer-events-none flex items-center gap-3 whitespace-nowrap transition-opacity duration-500"
      style={{ opacity: chapter < 0 ? 0 : 1 }}
    >
      <span className="font-mono text-[9.5px] tabular-nums text-ink-faint">
        {String(Math.max(0, chapter) + 1).padStart(2, "0")}
        <span className="text-ink-faint/50">/{String(CHAPTERS.length).padStart(2, "0")}</span>
      </span>
      <span className="flex items-center gap-1.5">
        {CHAPTERS.map((c, i) => (
          <span
            key={c.id}
            className={`h-px transition-all duration-500 ${
              i === chapter ? "w-8 bg-brand-bright" : "w-4 bg-white/15"
            }`}
          />
        ))}
      </span>
      <span className="font-mono text-[9.5px] uppercase tracking-[0.2em] text-brand-bright">
        {current?.label ?? ""}
      </span>
    </div>
  );
}

/* -------------------------------------------------------------------- rail */

export function HudRail({
  chapter,
  info,
  models,
  onLaunch,
  onSkip,
}: {
  chapter: number;
  info: ModelInfo | null;
  models: ModelSummary[];
  onLaunch: () => void;
  onSkip: () => void;
}) {
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 top-[52svh] z-20 flex flex-col lg:inset-y-0 lg:left-[60%] lg:right-0">
      <div className="flex h-full flex-col justify-start gap-5 overflow-hidden px-4 py-4 lg:justify-center lg:px-8 lg:py-8">
        <div className="hidden lg:block">
          <Progress chapter={chapter} />
        </div>

        {/* Centred below `lg`, where the rail is the full width of a stacked layout and a
            left-hugging panel leaves a column of dead space beside it. From `lg` up the rail
            is its own forty percent and the panel belongs at its left edge. */}
        <div className="mx-auto w-full max-w-[460px] lg:mx-0">
          <AnimatePresence mode="wait">
            {chapter === 0 && <HeroPanel key="hero" info={info} onLaunch={onLaunch} onSkip={onSkip} />}
            {chapter === 1 && <VectorPanel key="vectors" />}
            {chapter === 2 && <CorePanel key="core" models={models} />}
            {chapter === 3 && <VerdictPanel key="verdict" />}
            {chapter === 4 && <ConsolePanel key="console" onLaunch={onLaunch} onSkip={onSkip} />}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
