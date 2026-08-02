// The Tailwind theme colours as THREE.Color objects, so the 3D layer and the DOM stay
// visually identical instead of drifting into two different blues.
import { Color } from "three";

export const HEX = {
  brand: "#3987e5",
  brandBright: "#6aa9ff",
  fraud: "#e5484d",
  legit: "#30a46c",
  warn: "#f5b849",
  /** The card's metal, reused for the features the model actually leans on. */
  gold: "#d9ae55",
  ink: "#f4f5f6",
  inkMuted: "#a2a6ad",
  surface: "#111214",
  surfaceRaised: "#181a1d",
  line: "#26292e",
} as const;

export const COLOR = {
  brand: new Color(HEX.brand),
  fraud: new Color(HEX.fraud),
  legit: new Color(HEX.legit),
  warn: new Color(HEX.warn),
  gold: new Color(HEX.gold),
  ink: new Color(HEX.ink),
  surface: new Color(HEX.surface),
} as const;

const scratch = new Color();

/**
 * legit -> warn -> fraud ramp for a 0..1 risk score. Two segments rather than a straight
 * green->red lerp, because a direct interpolation passes through a muddy brown around 0.5.
 */
export function riskColor(t: number, target = scratch): Color {
  const clamped = Math.min(1, Math.max(0, t));
  if (clamped < 0.5) {
    return target.copy(COLOR.legit).lerp(COLOR.warn, clamped * 2);
  }
  return target.copy(COLOR.warn).lerp(COLOR.fraud, (clamped - 0.5) * 2);
}

/** Cool -> hot ramp used for the decision surface (blue = poor, amber = strong). */
export function surfaceColor(t: number, target = scratch): Color {
  const clamped = Math.min(1, Math.max(0, t));
  return target.copy(COLOR.brand).lerp(COLOR.warn, clamped);
}
