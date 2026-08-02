// State shared between the DOM heads-up display and the WebGL scenes.
//
// Same reasoning as scrollState: the confidence slider fires ~60 times a second while
// dragging, and the 3D reads it inside useFrame. Passing it as a React prop would rebuild
// the whole scene graph on every pixel of drag. The DOM keeps its own React state for what
// it renders; this object is the copy the renderer polls.
import { FALLBACK_CURVE, SweepPoint } from "../data/champion";
export const hud = {
  /** Decision cut-off, 0..1. Drives which path the verdict particles take. */
  threshold: 0.636,
  /** Feature groups the user has left switched on, by GROUPS[].id. */
  groups: { core: true, pca: true, raw: true } as Record<string, boolean>,
  /** Set while the user drags, so the gate can brace against it. */
  scrubbing: false,
  /**
   * Bumped by the console's inject button. The scene compares it against the value it last
   * acted on, which is a counter rather than a boolean so rapid clicks all land instead of
   * collapsing into one.
   */
  inject: 0,
};

/**
 * Probabilities of real held-out transactions, with the labels they actually carry.
 *
 * The label is what makes the verdict chapter able to tell the truth about its own
 * mistakes: a packet whose true label is fraud and whose probability falls below the
 * threshold is a fraud being waved through, and it is drawn as one.
 */
export const sampleRisk = { pool: [] as Array<{ p: number; fraud: boolean }> };

/**
 * Running tally of how the verdict scene has routed transactions. Written by the renderer
 * every time a packet clears the gate, and read by the HUD on a slow interval -- the DOM
 * has no business re-rendering once per packet.
 */
export const routed = { legit: 0, fraud: 0 };

/**
 * How badly the current threshold is failing, on the stream actually in front of you.
 *
 * Both are written by the verdict scene from the last few hundred packets it routed, and
 * both drive visible stress: the containment vault flares when it floods, and the whole
 * frame glitches when either climbs past about two thirds.
 *
 * They are two different failures at two different ends of the slider, and the scene shows
 * them as two different things because they are:
 *
 *   flood -- the threshold is low, so almost everything trips the alarm. The vault is full
 *            of legitimate customers. This is the false-alarm cost.
 *   leak  -- the threshold is high, so genuine fraud scores under it and is approved. Those
 *            packets are drawn crimson going down the emerald lane.
 */
export const alarm = { flood: 0, leak: 0 };

/**
 * How fast the camera is currently travelling down the corridor, normalised 0..1 against a
 * brisk transition.
 *
 * The point of the value is that the flight is one shot: the rushing streaks on the corridor
 * wall, the chromatic fringing and the bloom lift all key off actual camera speed, so the
 * picture reacts to travelling rather than to a scroll number that happens to correlate
 * with it.
 */
export const flight = { rush: 0, z: 0 };

/**
 * The champion's threshold sweep: 201 points of {threshold, TP, FP, FN, TN} over the whole
 * held-out test set.
 *
 * It starts as the copy transcribed into champion.ts and is replaced by the API's the moment
 * /model-info answers. Both the console's confusion pillars and the rail's readouts source
 * their numbers here, so the geometry and the text can never quote different figures for the
 * same threshold -- which they would the moment one of them fell back and the other did not.
 */
export const curve = { points: FALLBACK_CURVE as SweepPoint[] };
