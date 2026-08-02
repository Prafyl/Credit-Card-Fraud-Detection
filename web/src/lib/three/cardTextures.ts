// Procedural textures for the credit card.
//
// All three maps are drawn on a 2D canvas at module load rather than fetched, for the same
// reason the labels are (see labelTexture.ts): the demo has to survive a machine with no
// network. Nothing here needs an artist -- a card is a flat slab of brushed metal with
// stamped foil on it, and both of those are a few dozen lines of canvas work.

import { CanvasTexture, LinearFilter, RepeatWrapping, SRGBColorSpace, Texture } from "three";

/** Deterministic noise, so the card is byte-identical on every reload. */
function makeRandom(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function canvas2d(w: number, h: number): CanvasRenderingContext2D {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c.getContext("2d")!;
}

/* -------------------------------------------------------------------- glow */

let glow: Texture | null = null;

/**
 * A soft radial falloff, for light rather than for a surface.
 *
 * Additive geometry cannot do this. A circle of flat colour is uniformly bright right up to
 * its rim, so however low the opacity is set it reads as a disc pasted onto the shot -- and
 * bloom makes that worse, not better, by wrapping the hard edge in a halo. The falloff has
 * to be in the texture. Quartic rather than linear because a linear ramp still leaves a
 * visible edge where it reaches zero.
 */
export function glowSprite(): Texture {
  if (glow) return glow;

  const size = 256;
  const ctx = canvas2d(size, size);
  const image = ctx.createImageData(size, size);
  const half = size / 2;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const d = Math.hypot(x - half + 0.5, y - half + 0.5) / half;
      const a = Math.max(0, 1 - d);
      const falloff = a * a * a * a;
      const o = (y * size + x) * 4;
      // Warm at the core and cooling outward: a hot spot in gold with a paler corona, which
      // is what light through a small opening in metal actually looks like.
      image.data[o] = 255;
      image.data[o + 1] = 205 + falloff * 45;
      image.data[o + 2] = 130 + falloff * 90;
      image.data[o + 3] = Math.round(falloff * 255);
    }
  }

  ctx.putImageData(image, 0, 0);
  const texture = new CanvasTexture(ctx.canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  glow = texture;
  return texture;
}

/* ------------------------------------------------------------------ finish */

let brushed: { roughness: Texture; normal: Texture } | null = null;

/**
 * The card's surface finish: an anisotropic roughness map and a matching normal map.
 *
 * Both encode the same thing -- a horizontally brushed titanium face. The roughness map is
 * what makes the specular highlight stretch into a band instead of a dot, and the normal
 * map is what keeps the surface from reading as a flat photograph of metal when the card
 * tilts. Together they are the whole difference between "dark plastic" and "anodised metal".
 */
export function brushedFinish(): { roughness: Texture; normal: Texture } {
  if (brushed) return brushed;

  const W = 1024;
  const H = 640;

  /* roughness: mid-grey base, scratched with lighter and darker horizontal streaks */
  const r = canvas2d(W, H);
  r.fillStyle = "#4e4e4e"; // ~0.31 roughness
  r.fillRect(0, 0, W, H);

  const rand = makeRandom(0x5e17);
  for (let i = 0; i < 5200; i++) {
    const y = Math.floor(rand() * H);
    const x = rand() * W;
    const len = 40 + rand() * 320;
    const light = rand() > 0.5;
    r.strokeStyle = light ? `rgba(255,255,255,${0.03 + rand() * 0.05})` : `rgba(0,0,0,${0.03 + rand() * 0.05})`;
    r.lineWidth = rand() < 0.85 ? 1 : 2;
    r.beginPath();
    r.moveTo(x, y + 0.5);
    r.lineTo(x + len, y + 0.5);
    r.stroke();
  }

  // A very faint concentric guilloche, the pattern pressed into premium card stock. Almost
  // invisible head-on; it catches the light at a glancing angle, which is the point.
  r.save();
  r.globalAlpha = 0.05;
  r.strokeStyle = "#ffffff";
  r.lineWidth = 1;
  for (let k = 0; k < 46; k++) {
    r.beginPath();
    r.ellipse(W * 0.5, H * 0.5, 30 + k * 18, 18 + k * 11, 0, 0, Math.PI * 2);
    r.stroke();
  }
  r.restore();

  /* normal: flat (128,128,255) perturbed only across the brush direction */
  const n = canvas2d(W, H);
  const image = n.createImageData(W, H);
  const noise = makeRandom(0x1d0c);
  const rowBias = new Float32Array(H);
  for (let y = 0; y < H; y++) rowBias[y] = (noise() - 0.5) * 2;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      // Grooves run along X, so the surface slope lives almost entirely in Y.
      const groove = rowBias[y] * 14 + Math.sin(x * 0.11 + y * 1.7) * 3;
      image.data[i] = 128 + Math.sin(x * 0.31) * 2;
      image.data[i + 1] = 128 + groove;
      image.data[i + 2] = 255;
      image.data[i + 3] = 255;
    }
  }
  n.putImageData(image, 0, 0);

  const roughness = new CanvasTexture(r.canvas);
  roughness.wrapS = roughness.wrapT = RepeatWrapping;
  roughness.minFilter = LinearFilter;

  const normal = new CanvasTexture(n.canvas);
  normal.wrapS = normal.wrapT = RepeatWrapping;
  normal.minFilter = LinearFilter;

  brushed = { roughness, normal };
  return brushed;
}

/* ------------------------------------------------------------------- print */

/** Canvas has no letter-spacing before Chrome 99, and the eyebrows depend on it. */
function tracked(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  spacing: number,
  align: "left" | "right" = "left",
): void {
  const chars = [...text];
  const width = chars.reduce((w, c) => w + ctx.measureText(c).width + spacing, -spacing);
  let cursor = align === "right" ? x - width : x;
  for (const c of chars) {
    ctx.fillText(c, cursor, y);
    cursor += ctx.measureText(c).width + spacing;
  }
}

/**
 * Stamped foil: a dark impression under the glyph and a light catch above it, then the
 * glyph itself. Offsetting the two in opposite directions is what the eye reads as relief;
 * which direction is up is set by `deboss`.
 */
function stamp(
  ctx: CanvasRenderingContext2D,
  draw: (ctx: CanvasRenderingContext2D) => void,
  fill: string | CanvasGradient,
  deboss = false,
): void {
  const d = deboss ? -1 : 1;

  ctx.save();
  ctx.translate(3 * d, 4 * d);
  ctx.fillStyle = "rgba(0,0,0,0.62)";
  draw(ctx);
  ctx.restore();

  ctx.save();
  ctx.translate(-2 * d, -2.5 * d);
  ctx.fillStyle = "rgba(255,255,255,0.26)";
  draw(ctx);
  ctx.restore();

  ctx.save();
  ctx.fillStyle = fill;
  draw(ctx);
  ctx.restore();
}

export interface CardPrint {
  /** Cardholder name, stamped in gold along the bottom edge. */
  holder: string;
  /** Primary account number, already masked. */
  pan: string;
  expiry: string;
  brand: string;
  tagline: string;
}

let printCache: Texture | null = null;

/**
 * Everything printed on the card face, baked into one texture.
 *
 * One texture rather than a sprite per string, for three reasons: sprites are billboards
 * and would swivel to face the camera while the card tilts, which instantly breaks the
 * illusion; the relief has to be baked per-pixel and a sprite cannot carry it; and it is
 * one draw call instead of eight.
 *
 * The chip and the contactless mark are real geometry and are deliberately left out -- the
 * empty band across the upper left is where they sit.
 */
export function cardPrintTexture(print: CardPrint): Texture {
  if (printCache) return printCache;

  const W = 2048;
  const H = 1292; // 85.60 x 53.98 mm, ISO/IEC 7810 ID-1
  const ctx = canvas2d(W, H);
  ctx.textBaseline = "alphabetic";

  const silver = () => {
    const g = ctx.createLinearGradient(0, 760, 0, 900);
    g.addColorStop(0, "#f2f5f8");
    g.addColorStop(0.45, "#b9c1cc");
    g.addColorStop(0.55, "#e8edf3");
    g.addColorStop(1, "#8f98a5");
    return g;
  };

  const gold = (top: number, bottom: number) => {
    const g = ctx.createLinearGradient(0, top, 0, bottom);
    g.addColorStop(0, "#f6e3ad");
    g.addColorStop(0.42, "#d9ae55");
    g.addColorStop(0.58, "#b8862c");
    g.addColorStop(1, "#efd79c");
    return g;
  };

  /* --- issuer mark, top right ------------------------------------------- */
  ctx.font = "600 72px Inter, 'Segoe UI', system-ui, sans-serif";
  ctx.fillStyle = "#e6ebf2";
  tracked(ctx, print.brand, W - 148, 214, 11, "right");

  ctx.font = "500 33px 'JetBrains Mono', Consolas, monospace";
  ctx.fillStyle = "rgba(186,199,215,0.86)";
  tracked(ctx, print.tagline, W - 150, 270, 3, "right");

  /* --- primary account number ------------------------------------------- */
  ctx.font = "700 116px 'JetBrains Mono', Consolas, monospace";
  stamp(ctx, (c) => tracked(c, print.pan, 152, 852, 7), silver());

  /* --- expiry, on its own two lines below the number --------------------- */
  ctx.font = "600 25px 'JetBrains Mono', Consolas, monospace";
  ctx.fillStyle = "rgba(158,171,188,0.8)";
  tracked(ctx, "VALID THRU", 154, 934, 6);

  ctx.font = "700 58px 'JetBrains Mono', Consolas, monospace";
  stamp(ctx, (c) => tracked(c, print.expiry, 152, 1000, 6), "#c6ced9");

  /* --- cardholder -------------------------------------------------------- */
  ctx.font = "700 80px Inter, 'Segoe UI', system-ui, sans-serif";
  stamp(
    ctx,
    (c) => tracked(c, print.holder, 152, 1158, 12),
    gold(1090, 1168),
    true, // pressed into the card, not raised off it
  );

  /* --- mark, bottom right ------------------------------------------------ */
  // A hexagonal aperture with an inward chevron. Deliberately abstract: imitating a real
  // scheme's mark on a demo card would be passing off someone else's brand.
  const cx = W - 232;
  const cy = 1052;
  const r = 96;

  ctx.save();
  ctx.strokeStyle = "rgba(217,174,85,0.85)";
  ctx.lineWidth = 6;
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = (Math.PI / 3) * i - Math.PI / 2;
    const px = cx + Math.cos(a) * r;
    const py = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.stroke();

  ctx.strokeStyle = "rgba(240,244,249,0.9)";
  ctx.lineWidth = 11;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(cx - 34, cy - 14);
  ctx.lineTo(cx, cy + 26);
  ctx.lineTo(cx + 34, cy - 40);
  ctx.stroke();
  ctx.restore();

  const texture = new CanvasTexture(ctx.canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.anisotropy = 8;
  printCache = texture;
  return texture;
}

/** Frees the module-level caches. Only the dev server's hot reload needs this. */
export function disposeCardTextures(): void {
  brushed?.roughness.dispose();
  brushed?.normal.dispose();
  printCache?.dispose();
  glow?.dispose();
  brushed = null;
  printCache = null;
  glow = null;
}
