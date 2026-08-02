// A numeric readout that lives in the 3D scene and changes while you watch it.
//
// <Label> caches a texture per string, which is right for names that never change and wrong
// for a number that changes sixty times a second -- it would allocate a canvas and a GPU
// texture per frame. This draws into one canvas of a fixed size and re-uploads it only when
// the text it is asked to show actually differs, which for a value formatted to two decimals
// is a handful of times a second rather than sixty.
//
// It is deliberately laid out like an instrument channel: a caption in small caps on the
// left, the value right-aligned in tabular mono, a hairline rule under both. Numbers that
// move need a fixed frame to move inside, or the eye tracks the jitter instead of the value.
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import { CanvasTexture, LinearFilter, NormalBlending, SRGBColorSpace, SpriteMaterial } from "three";

export interface ReadoutHandle {
  /** Sets the displayed value. A no-op if the string is unchanged. */
  set(value: string): void;
}

const WIDTH = 460;
const HEIGHT = 132;

function draw(ctx: CanvasRenderingContext2D, caption: string, value: string, color: string) {
  ctx.clearRect(0, 0, WIDTH, HEIGHT);

  const inset = 4;
  ctx.fillStyle = "rgba(8, 10, 15, 0.9)";
  ctx.fillRect(inset, inset, WIDTH - inset * 2, HEIGHT - inset * 2);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.09)";
  ctx.lineWidth = 2;
  ctx.strokeRect(inset, inset, WIDTH - inset * 2, HEIGHT - inset * 2);

  // Corner brackets in the accent, matching the DOM rail's panels. Same instrument, two
  // different surfaces.
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  const b = 22;
  for (const [x, dx] of [
    [inset, 1],
    [WIDTH - inset, -1],
  ] as const) {
    for (const [y, dy] of [
      [inset, 1],
      [HEIGHT - inset, -1],
    ] as const) {
      ctx.beginPath();
      ctx.moveTo(x + dx * b, y);
      ctx.lineTo(x, y);
      ctx.lineTo(x, y + dy * b);
      ctx.stroke();
    }
  }

  ctx.textBaseline = "middle";
  ctx.font = '600 26px "JetBrains Mono", Consolas, monospace';
  ctx.fillStyle = "rgba(162, 166, 173, 0.95)";
  ctx.textAlign = "left";
  ctx.fillText(caption, 30, 40);

  ctx.font = '700 58px "JetBrains Mono", Consolas, monospace';
  ctx.textAlign = "right";
  ctx.shadowColor = color;
  ctx.shadowBlur = 22;
  ctx.fillStyle = color;
  ctx.fillText(value, WIDTH - 30, 88);
  ctx.shadowBlur = 0;
}

export const Readout = forwardRef<
  ReadoutHandle,
  {
    caption: string;
    initial: string;
    position: [number, number, number];
    /** World height of the panel; width follows from the canvas aspect. */
    height?: number;
    color?: string;
  }
>(function Readout({ caption, initial, position, height = 0.5, color = "#6aa9ff" }, ref) {
  const shown = useRef(initial);

  const { texture, context } = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    const ctx = canvas.getContext("2d")!;
    draw(ctx, caption, initial, color);

    const map = new CanvasTexture(canvas);
    map.colorSpace = SRGBColorSpace;
    map.minFilter = LinearFilter;
    map.magFilter = LinearFilter;
    return { texture: map, context: ctx };
  }, [caption, initial, color]);

  const material = useMemo(
    () =>
      new SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        // The panel carries its own dark ground, so it composites normally rather than
        // adding itself to whatever it happens to be in front of.
        blending: NormalBlending,
      }),
    [texture],
  );

  useImperativeHandle(
    ref,
    () => ({
      set(value: string) {
        if (value === shown.current) return;
        shown.current = value;
        draw(context, caption, value, color);
        texture.needsUpdate = true;
      },
    }),
    [caption, color, context, texture],
  );

  useEffect(
    () => () => {
      material.dispose();
      texture.dispose();
    },
    [material, texture],
  );

  return (
    <sprite position={position} scale={[(height * WIDTH) / HEIGHT, height, 1]} renderOrder={24}>
      <primitive object={material} attach="material" />
    </sprite>
  );
});
