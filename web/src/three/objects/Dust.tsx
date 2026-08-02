// Ambient particulate, filling the whole corridor from just behind the card to past the
// console. It is the cheapest possible parallax reference: without something at a known
// distance drifting past, a camera flying down a dark tunnel has nothing to be measured
// against and the flight reads as a zoom.
//
// It is deliberately absent from chapter 1. A starfield behind a floating credit card reads
// as science fiction; the card has to be photographed against nothing, and the space only
// arrives once the camera is through the chip.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { AdditiveBlending, BufferGeometry, Float32BufferAttribute, Points, PointsMaterial } from "three";
import { HEX } from "../../lib/three/palette";
import { scrollProgress } from "../../lib/three/scrollState";
import { CARD_ASPECT, CORRIDOR_END, corridorHalfWidth, ease } from "../chapters";

export function Dust({ count = 1400 }: { count?: number }) {
  const ref = useRef<Points>(null);

  const geometry = useMemo(() => {
    const positions = new Float32Array(count * 3);
    let seed = 1337;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296;
      return seed / 4294967296;
    };
    for (let i = 0; i < count; i++) {
      const z = -2 + rand() * (CORRIDOR_END + 2);
      // Placed against the local width of the corridor rather than in a fixed cylinder, so
      // the density on screen stays even as the tunnel opens out.
      const half = corridorHalfWidth(z);
      const angle = rand() * Math.PI * 2;
      // sqrt keeps the area density uniform instead of clumping everything on the axis.
      const radius = half * (0.18 + Math.sqrt(rand()) * 1.05);
      positions[i * 3 + 0] = Math.cos(angle) * radius;
      positions[i * 3 + 1] = (Math.sin(angle) * radius) / CARD_ASPECT;
      positions[i * 3 + 2] = z;
    }
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(positions, 3));
    return g;
  }, [count]);

  const material = useMemo(
    () =>
      new PointsMaterial({
        color: HEX.brand,
        size: 0.035,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.3,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  useEffect(
    () => () => {
      geometry.dispose();
      material.dispose();
    },
    [geometry, material],
  );

  useFrame((state) => {
    if (!ref.current) return;
    // A slow roll rather than translation: motion parallax against a moving camera is
    // stronger than any drift speed worth using, and rolling costs one matrix update.
    ref.current.rotation.z = state.clock.elapsedTime * 0.012;
    // Arrives with the chip dissolve and stays for the rest of the story.
    material.opacity = 0.24 * ease(scrollProgress.current, 0.155, 0.275);
    ref.current.visible = material.opacity > 0.004;
  });

  return <points ref={ref} geometry={geometry} material={material} frustumCulled={false} />;
}
