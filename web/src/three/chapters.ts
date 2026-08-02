// The scroll timeline, in one place.
//
// The page is one continuous flight down a single corridor, not five separate scenes. Every
// chapter is built around its own anchor on the -Z axis of one shared world, and the camera
// flies from the first to the last without ever cutting. Chapter windows *overlap* on
// purpose: for the length of the overlap you are inside both, which is what makes the
// handover read as travelling through a layer rather than as one slide replacing another.
//
// That is the whole difference from the previous build, which faded each chapter to nothing
// before the next one arrived and therefore felt like four sections wearing a scroll bar.
//
// Distance does the work that opacity used to. The scene is fogged at a density tuned to the
// spacing below, so a chapter behind you dims because it is behind you. The presence value
// here only exists to stop the far end of the corridor from being drawn before it can
// possibly matter, and to guarantee a hard zero at the edges so nothing pops when a chapter
// is culled.
//
// The DOM panels in the right-hand rail read the same numbers, so text and geometry change
// together rather than drifting apart.

export type ChapterId = "card" | "vectors" | "core" | "verdict" | "console";

export interface Chapter {
  id: ChapterId;
  /** Shown in the rail's progress strip. */
  label: string;
  /**
   * Where the chapter is built, on the corridor axis. Scenes are authored around their own
   * origin and <Chapter> puts them here, so no scene has to know about any other.
   */
  z: number;
  /** Scroll window over which any of the chapter's geometry may be on screen. */
  span: [number, number];
  /** Scroll window over which the rail holds this chapter's panel and the camera dwells. */
  hold: [number, number];
}

/**
 * Scroll distance over which a chapter's geometry ramps to full and back.
 *
 * Short on purpose. By the time a chapter is fading it is already tens of units behind the
 * camera and most of the way into the fog; this ramp is a safety net that guarantees a clean
 * zero at the cull boundary, not the transition itself.
 */
const FADE = 0.045;

// Spans that start before 0 or end after 1 are how the first and last chapters avoid ramping
// at the ends of the page -- the story opens on the card and closes on the console, and
// neither should fade at a boundary the reader can reach.
export const CHAPTERS: Chapter[] = [
  { id: "card", label: "The card", z: 0, span: [-0.06, 0.235], hold: [0.0, 0.135] },
  { id: "vectors", label: "Feature space", z: -34, span: [0.105, 0.475], hold: [0.215, 0.375] },
  { id: "core", label: "Decision core", z: -76, span: [0.355, 0.7], hold: [0.455, 0.605] },
  { id: "verdict", label: "The verdict", z: -118, span: [0.575, 0.925], hold: [0.675, 0.845] },
  { id: "console", label: "The console", z: -152, span: [0.835, 1.08], hold: [0.895, 1.0] },
];

/** Midpoints between consecutive holds: where the rail swaps one panel for the next. */
const BOUNDARIES = [0.175, 0.415, 0.64, 0.87];

/** Index of the chapter that owns scroll position `p`. */
export function chapterAt(p: number): number {
  for (let i = BOUNDARIES.length - 1; i >= 0; i--) if (p >= BOUNDARIES[i]) return i + 1;
  return 0;
}

/** Remap `p` from the range [a, b] onto 0..1, clamped outside it. */
export function range(p: number, a: number, b: number): number {
  if (b === a) return p >= b ? 1 : 0;
  return Math.min(1, Math.max(0, (p - a) / (b - a)));
}

/** Smoothstep-eased `range`, for anything that should not start or stop abruptly. */
export function ease(p: number, a: number, b: number): number {
  const t = range(p, a, b);
  return t * t * (3 - 2 * t);
}

/** How present chapter `c` is at scroll `p`: 0 offstage, 1 fully in the corridor. */
export function presenceOf(c: Chapter, p: number): number {
  return Math.min(ease(p, c.span[0], c.span[0] + FADE), 1 - ease(p, c.span[1] - FADE, c.span[1]));
}

/**
 * 0..1 progress *through* a chapter's held span, ignoring the flight in and the flight out.
 *
 * Scenes use this for motion that should read as continuous while the chapter is the subject
 * of the shot -- a core turning, a cluster opening -- as distinct from arrival and
 * departure, which the camera path is already expressing by moving.
 */
export function localProgress(c: Chapter, p: number): number {
  return range(p, c.hold[0], c.hold[1]);
}

/** Where the card's chip sits, in card object space. The camera aims here, the card burns
 *  open from here, and the corridor's first frame is centred on it, so all three agree. */
export const CHIP: [number, number] = [-1.02, 0.3];

/** Far end of the corridor, a little past the last chapter. Fixes the fog and dust extents. */
export const CORRIDOR_END = -168;

/**
 * Half-width of the corridor at depth `z`.
 *
 * The tunnel starts at the width of the card and opens out as the subject matter does: a
 * payment is one object, a feature space is a cluster, an ensemble is a machine. Both the
 * framing lattice and the rushing streaks read this, so the walls stay one surface.
 */
export function corridorHalfWidth(z: number): number {
  const t = Math.min(1, Math.max(0, z / CORRIDOR_END));
  return 1.78 + t * 15.9;
}

/** The card's own proportion, ISO/IEC 7810 ID-1. The corridor keeps it the whole way down. */
export const CARD_ASPECT = 3.424 / 2.159;
