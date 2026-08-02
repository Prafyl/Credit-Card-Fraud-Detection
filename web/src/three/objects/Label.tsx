// A camera-facing text label. Wraps labelTexture in a sprite so callers place text by
// world position and height, without thinking about pixel dimensions or aspect ratio.
import { useEffect, useMemo } from "react";
import { AdditiveBlending, NormalBlending, SpriteMaterial } from "three";
import { labelAspect, labelTexture, LabelOptions } from "../../lib/three/labelTexture";

export function Label({
  text,
  position,
  height = 0.22,
  opacity = 1,
  renderOrder = 10,
  ...opts
}: {
  text: string;
  position: [number, number, number];
  /** World height of the badge; width follows from the rasterised aspect ratio. */
  height?: number;
  opacity?: number;
  renderOrder?: number;
} & LabelOptions) {
  const texture = useMemo(
    () => labelTexture(text, opts),
    // The option fields are primitives, so listing them keeps the memo exact without a
    // deep compare on the spread object.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [text, opts.color, opts.weight, opts.size, opts.mono, opts.pill, opts.tracking],
  );

  const material = useMemo(
    () =>
      new SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        // A badge carries its own dark background and has to survive being drawn over a
        // bright node, so it composites normally and ignores depth. Bare glowing text has
        // nothing to occlude and is additive, which lets bloom pick it up.
        blending: opts.pill ? NormalBlending : AdditiveBlending,
        depthTest: !opts.pill,
        opacity,
      }),
    [texture, opts.pill, opacity],
  );

  useEffect(() => () => material.dispose(), [material]);

  const aspect = labelAspect(texture);

  return (
    <sprite
      position={position}
      scale={[height * aspect, height, 1]}
      renderOrder={opts.pill ? renderOrder + 10 : renderOrder}
    >
      <primitive object={material} attach="material" />
    </sprite>
  );
}
