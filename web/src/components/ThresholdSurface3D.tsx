// Interactive version of the decision surface, embedded in the dashboard.
//
// The key interaction: the DOM threshold slider already in the dashboard drives the
// translucent slicing plane here. Moving the slider walks the plane along the surface, so
// the 2D control gains a 3D readout without introducing any new state.
import { Html, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef, useState } from "react";
import { DoubleSide, Mesh } from "three";
import { easing } from "maath";
import { CurvePoint } from "../api";
import { buildSurfaceGrid, curveMetrics } from "../lib/data/curveMath";
import { HEX } from "../lib/three/palette";
import { canRender3D, gpuTier } from "../lib/three/capabilities";
import {
  SURFACE_HEIGHT,
  SURFACE_WIDTH,
  SurfaceMesh,
} from "../three/objects/SurfaceMesh";

function SlicePlane({ threshold }: { threshold: number }) {
  const ref = useRef<Mesh>(null);
  useFrame((_, delta) => {
    if (!ref.current) return;
    const targetX = (threshold - 0.5) * SURFACE_WIDTH;
    easing.damp(ref.current.position, "x", targetX, 0.25, delta);
  });
  return (
    <mesh ref={ref} rotation={[0, Math.PI / 2, 0]} position={[0, SURFACE_HEIGHT / 2, 0]}>
      <planeGeometry args={[8, SURFACE_HEIGHT * 1.6]} />
      <meshBasicMaterial
        color={HEX.warn}
        transparent
        opacity={0.18}
        side={DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function SurfaceScene({
  curves,
  order,
  threshold,
}: {
  curves: Map<string, CurvePoint[]>;
  order: string[];
  threshold: number;
}) {
  const grid = useMemo(
    () => (curves.size > 0 ? buildSurfaceGrid(curves, order) : null),
    [curves, order],
  );

  return (
    <>
      <SurfaceMesh grid={grid} animate />
      <SlicePlane threshold={threshold} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[6, 10, 6]} intensity={1.6} />
      <pointLight position={[-8, 4, -4]} intensity={35} color={HEX.brand} distance={30} />
      <OrbitControls
        enablePan={false}
        minPolarAngle={0.15 * Math.PI}
        maxPolarAngle={0.48 * Math.PI}
        minDistance={9}
        maxDistance={22}
        autoRotate
        autoRotateSpeed={0.35}
      />
    </>
  );
}

export function ThresholdSurface3D({
  curves,
  order,
  threshold,
  model,
}: {
  curves: Map<string, CurvePoint[]>;
  order: string[];
  threshold: number;
  model: string;
}) {
  const [enabled] = useState(() => canRender3D());
  const tier = gpuTier();

  // Numbers for the caption come from the same curve the surface is built from, so the
  // label can never disagree with the geometry.
  const current = useMemo(() => {
    const curve = curves.get(model);
    if (!curve) return null;
    const metrics = curveMetrics(curve);
    return metrics.reduce((best, p) =>
      Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best,
    );
  }, [curves, model, threshold]);

  if (!enabled || curves.size === 0) {
    return (
      <div className="grid h-[420px] place-items-center rounded-2xl border border-line bg-surface-inset text-sm text-ink-faint">
        {curves.size === 0
          ? "Loading model curves…"
          : "3D view unavailable (reduced motion or no WebGL)"}
      </div>
    );
  }

  return (
    <div className="relative h-[420px] overflow-hidden rounded-2xl border border-line bg-surface-inset">
      <Canvas
        dpr={[1, tier === "low" ? 1 : 1.6]}
        gl={{ antialias: true, powerPreference: "high-performance" }}
        camera={{ position: [11, 7, 11], fov: 45 }}
      >
        <Suspense fallback={<Html center className="text-xs text-ink-faint">Building surface…</Html>}>
          <SurfaceScene curves={curves} order={order} threshold={threshold} />
        </Suspense>
      </Canvas>

      <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-line bg-surface/80 px-3 py-2 text-xs backdrop-blur">
        <div className="text-ink-faint">Height = F1 · Colour = precision</div>
        {current && (
          <div className="mt-1 font-mono text-ink">
            t={current.threshold.toFixed(3)} · P={current.precision.toFixed(3)} · R=
            {current.recall.toFixed(3)}
          </div>
        )}
      </div>
      <div className="pointer-events-none absolute bottom-4 right-4 text-xs text-ink-faint">
        drag to orbit · {order.length} models × 201 thresholds
      </div>
    </div>
  );
}
