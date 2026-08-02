// Feature detection for the 3D layer. Every scene is optional decoration over data that is
// also shown in plain DOM, so anything that cannot render 3D must degrade rather than break.

let cached: boolean | null = null;

/** True if the browser can give us a WebGL context at all. Result is cached. */
export function hasWebGL(): boolean {
  if (cached !== null) return cached;
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2") ??
      canvas.getContext("webgl") ??
      canvas.getContext("experimental-webgl");
    cached = gl != null;
  } catch {
    cached = false;
  }
  return cached;
}

/** Honour the OS "reduce motion" setting, matching the convention already in index.css. */
export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * Rough capability tier from the number of logical cores and device memory. Used to scale
 * particle counts and pixel ratio so a laptop iGPU is not asked to do the same work as a
 * discrete card.
 */
export function gpuTier(): "low" | "mid" | "high" {
  if (typeof navigator === "undefined") return "mid";
  const cores = navigator.hardwareConcurrency ?? 4;
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 4;
  if (cores <= 4 || memory <= 4) return "low";
  if (cores <= 8) return "mid";
  return "high";
}

/** Should the 3D layer run at all? */
export function canRender3D(): boolean {
  return hasWebGL() && !prefersReducedMotion();
}
