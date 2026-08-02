// The 8 models as a 4x2 field of 3D bars: 4 algorithms across, 2 imbalance strategies deep.
// A single InstancedMesh, so all eight bars cost one draw call.
import { useFrame } from "@react-three/fiber";
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { Color, InstancedMesh, MeshStandardMaterial, Object3D } from "three";
import { ModelSummary } from "../../api";
import { surfaceColor } from "../../lib/three/palette";

const ALGORITHMS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"];
const STRATEGIES = ["weighted", "SMOTE"];
const SPACING_X = 2.0;
const SPACING_Z = 2.0;
const MAX_HEIGHT = 4.0;

interface Cell {
  index: number;
  model: ModelSummary | undefined;
  x: number;
  z: number;
}

export function MetricBars({
  models,
  active,
  onHover,
}: {
  models: ModelSummary[];
  active?: string | null;
  onHover?: (name: string | null) => void;
}) {
  const meshRef = useRef<InstancedMesh>(null);
  const dummy = useMemo(() => new Object3D(), []);
  const growth = useRef(0);
  const [hovered, setHovered] = useState<number | null>(null);

  const cells = useMemo<Cell[]>(() => {
    const out: Cell[] = [];
    let i = 0;
    STRATEGIES.forEach((strategy, zi) => {
      ALGORITHMS.forEach((algo, xi) => {
        const name = `${algo} + ${strategy}`;
        out.push({
          index: i++,
          model: models.find((m) => m.model_name === name),
          x: (xi - (ALGORITHMS.length - 1) / 2) * SPACING_X,
          z: (zi - (STRATEGIES.length - 1) / 2) * SPACING_Z,
        });
      });
    });
    return out;
  }, [models]);

  const material = useMemo(
    () => new MeshStandardMaterial({ metalness: 0.3, roughness: 0.45, vertexColors: false }),
    [],
  );

  // Instance colours are set once; only the per-frame transform changes during the grow-in.
  useLayoutEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const c = new Color();
    cells.forEach((cell) => {
      const pr = cell.model?.test_metrics.PR_AUC ?? 0;
      mesh.setColorAt(cell.index, surfaceColor(pr, c));
    });
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [cells]);

  useFrame((_, delta) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    growth.current = Math.min(1, growth.current + delta * 0.9);
    const ease = 1 - Math.pow(1 - growth.current, 3);

    cells.forEach((cell) => {
      const pr = cell.model?.test_metrics.PR_AUC ?? 0;
      const isActive = active != null && cell.model?.model_name === active;
      // Hover lifts a bar; the globally selected model stays permanently raised and wider
      // so you can see which one the rest of the dashboard is describing.
      const lift = cell.index === hovered ? 1.08 : 1;
      const h = Math.max(0.02, pr * MAX_HEIGHT * ease * lift);
      const width = isActive ? 1.02 : 0.9;
      dummy.position.set(cell.x, h / 2, cell.z);
      dummy.scale.set(width, h, width);
      dummy.updateMatrix();
      mesh.setMatrixAt(cell.index, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, cells.length]}
      material={material}
      onPointerMove={(e) => {
        const id = e.instanceId ?? null;
        setHovered(id);
        onHover?.(id != null ? (cells[id]?.model?.model_name ?? null) : null);
      }}
      onPointerOut={() => {
        setHovered(null);
        onHover?.(null);
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
    </instancedMesh>
  );
}

export { ALGORITHMS, STRATEGIES, SPACING_X, SPACING_Z, MAX_HEIGHT };
export type { Cell };
