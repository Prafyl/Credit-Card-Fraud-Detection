// Text labels in 3D, drawn on a 2D canvas and used as a sprite texture.
//
// Two alternatives were rejected. drei's <Text> is built on troika, which fetches its
// default Roboto font from fonts.gstatic.com at runtime -- an external request that turns
// into blank labels the moment the machine is offline, which is exactly the situation a
// classroom demo runs in. drei's <Html> renders real DOM, but it writes a transform to
// every element on every frame, and there are thirty-one of these.
//
// A canvas texture uses the system font stack, needs no network, and renders as one sprite
// with no per-frame DOM work.
import { CanvasTexture, LinearFilter, SRGBColorSpace, Texture } from "three";

const cache = new Map<string, Texture>();

export interface LabelOptions {
  color?: string;
  weight?: number;
  size?: number;
  mono?: boolean;
  /**
   * Draw the text inside a dark rounded badge outlined in `color`.
   *
   * Bare glowing text is legible over empty space and illegible the moment a bright node
   * drifts behind it. A badge carries its own background, so a label stays readable
   * wherever the scene happens to put it.
   */
  pill?: boolean;
  /** Letter-spacing in pixels at the rasterised size. */
  tracking?: number;
}

/** Canvas has no letter-spacing before Chrome 99, and the badges depend on it. */
function tracked(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, spacing: number): number {
  const chars = [...text];
  const width = chars.reduce((w, c) => w + ctx.measureText(c).width + spacing, -spacing);
  let cursor = x - width / 2;
  for (const c of chars) {
    ctx.fillText(c, cursor + ctx.measureText(c).width / 2, y);
    cursor += ctx.measureText(c).width + spacing;
  }
  return width;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Hex to `rgba(...)`, so one accent colour can drive fill, stroke and glow at three alphas. */
function rgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

/**
 * Rasterises `text` at a fixed device scale and returns a cached texture. Labels repeat a
 * lot across the scenes (feature names appear on nodes and again on connections), so the
 * cache keeps the count of GPU textures near the count of *distinct* strings.
 */
export function labelTexture(text: string, opts: LabelOptions = {}): Texture {
  const {
    color = "#f4f5f6",
    weight = 600,
    size = 64,
    mono = true,
    pill = false,
    tracking = mono ? size * 0.06 : size * 0.03,
  } = opts;

  const key = `${text}|${color}|${weight}|${size}|${mono}|${pill}|${tracking}`;
  const hit = cache.get(key);
  if (hit) return hit;

  const family = mono
    ? '"JetBrains Mono", Consolas, monospace'
    : 'Inter, "Segoe UI", system-ui, sans-serif';
  const font = `${weight} ${size}px ${family}`;

  // Measure on a scratch context first so the canvas is only as wide as the glyphs need.
  const scratch = document.createElement("canvas").getContext("2d")!;
  scratch.font = font;
  const glyphs = [...text].reduce((w, c) => w + scratch.measureText(c).width + tracking, -tracking);

  const padX = pill ? size * 0.62 : size * 0.34;
  const padY = pill ? size * 0.42 : size * 0.3;
  const width = Math.ceil(glyphs + padX * 2);
  const height = Math.ceil(size + padY * 2);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.font = font;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  if (pill) {
    const inset = 3;
    const r = (height - inset * 2) / 2;

    // Outer bloom, so the badge separates from whatever is behind it before the stroke does.
    ctx.save();
    ctx.shadowColor = rgba(color, 0.55);
    ctx.shadowBlur = size * 0.55;
    ctx.fillStyle = "rgba(7, 9, 13, 0.9)";
    roundRect(ctx, inset, inset, width - inset * 2, height - inset * 2, r);
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = "rgba(10, 13, 18, 0.88)";
    roundRect(ctx, inset, inset, width - inset * 2, height - inset * 2, r);
    ctx.fill();

    ctx.strokeStyle = rgba(color, 0.72);
    ctx.lineWidth = Math.max(2, size * 0.045);
    roundRect(ctx, inset, inset, width - inset * 2, height - inset * 2, r);
    ctx.stroke();

    ctx.fillStyle = color;
    tracked(ctx, text, width / 2, height / 2 + size * 0.02, tracking);
  } else {
    // A soft copy of the glyphs underneath gives the label its own bloom seed, so it reads
    // as emissive rather than as a flat decal pasted onto the scene.
    ctx.shadowColor = color;
    ctx.shadowBlur = size * 0.45;
    ctx.fillStyle = color;
    tracked(ctx, text, width / 2, height / 2, tracking);
    ctx.shadowBlur = 0;
    tracked(ctx, text, width / 2, height / 2, tracking);
  }

  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.needsUpdate = true;
  // Aspect ratio travels with the texture so callers can size a sprite without re-measuring.
  texture.userData.aspect = width / height;
  cache.set(key, texture);
  return texture;
}

/** Width/height of a rasterised label, for scaling the sprite that carries it. */
export function labelAspect(texture: Texture): number {
  return (texture.userData.aspect as number) ?? 4;
}

export function disposeLabelCache(): void {
  for (const texture of cache.values()) texture.dispose();
  cache.clear();
}
