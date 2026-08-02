// The 4x2 model grid as 3D bars. Hovering a bar names it and shows its PR-AUC.
import { Html, OrbitControls, Text } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Suspense, useState } from "react";
import { ModelSummary } from "../api";
import { canRender3D, gpuTier } from "../lib/three/capabilities";
import { HEX } from "../lib/three/palette";
import {
  ALGORITHMS,
  MetricBars,
  SPACING_X,
  SPACING_Z,
  STRATEGIES,
} from "../three/objects/MetricBars";

function Labels() {
  return (
    <>
      {ALGORITHMS.map((algo, i) => (
        <Text
          key={algo}
          position={[(i - (ALGORITHMS.length - 1) / 2) * SPACING_X, 0.02, 2.4]}
          rotation={[-Math.PI / 2, 0, 0]}
          fontSize={0.24}
          color={HEX.inkMuted}
          anchorX="center"
        >
          {algo === "LogisticRegression" ? "LogReg" : algo === "RandomForest" ? "Forest" : algo}
        </Text>
      ))}
      {STRATEGIES.map((s, i) => (
        <Text
          key={s}
          position={[-4.6, 0.02, (i - (STRATEGIES.length - 1) / 2) * SPACING_Z]}
          rotation={[-Math.PI / 2, 0, 0]}
          fontSize={0.24}
          color={HEX.inkMuted}
          anchorX="center"
        >
          {s}
        </Text>
      ))}
    </>
  );
}

export function MetricsBarField3D({
  models,
  active,
}: {
  models: ModelSummary[];
  active?: string | null;
}) {
  const [enabled] = useState(() => canRender3D());
  const [hovered, setHovered] = useState<string | null>(null);
  const tier = gpuTier();

  if (!enabled || models.length === 0) {
    return (
      <div className="grid h-[420px] place-items-center rounded-2xl border border-line bg-surface-inset text-sm text-ink-faint">
        {models.length === 0 ? "Loading models…" : "3D view unavailable"}
      </div>
    );
  }

  const shown = models.find((m) => m.model_name === (hovered ?? active));

  return (
    <div className="relative h-[420px] overflow-hidden rounded-2xl border border-line bg-surface-inset">
      <Canvas
        dpr={[1, tier === "low" ? 1 : 1.6]}
        gl={{ antialias: true }}
        camera={{ position: [6.5, 6, 8], fov: 45 }}
      >
        <Suspense fallback={<Html center className="text-xs text-ink-faint">Loading…</Html>}>
          <MetricBars models={models} active={active} onHover={setHovered} />
          <Labels />
          <gridHelper args={[12, 12, HEX.line, HEX.line]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[5, 10, 5]} intensity={1.7} />
          <pointLight position={[-6, 5, -3]} intensity={30} color={HEX.brand} distance={26} />
          <OrbitControls
            enablePan={false}
            minPolarAngle={0.1 * Math.PI}
            maxPolarAngle={0.47 * Math.PI}
            minDistance={7}
            maxDistance={18}
            autoRotate
            autoRotateSpeed={0.5}
          />
        </Suspense>
      </Canvas>

      <div className="pointer-events-none absolute left-4 top-4 rounded-lg border border-line bg-surface/80 px-3 py-2 text-xs backdrop-blur">
        <div className="text-ink-faint">Bar height = PR-AUC</div>
        {shown && (
          <div className="mt-1">
            <div className="font-medium text-ink">{shown.model_name}</div>
            <div className="font-mono text-ink-muted">
              PR-AUC {shown.test_metrics.PR_AUC.toFixed(3)} · recall{" "}
              {shown.test_metrics.recall.toFixed(3)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
