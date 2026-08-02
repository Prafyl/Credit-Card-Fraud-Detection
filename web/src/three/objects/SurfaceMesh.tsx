// The decision surface: every model's threshold sweep as one continuous 3D landscape.
//
// x = threshold (0 -> 1), z = model, y = F1 at that operating point, colour = precision.
// This is real data straight from the API's threshold_curve, not a generated shape -- the
// ridge running along the surface IS each model's usable operating range, and the cliff at
// low thresholds is where precision collapses under false alarms.
import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import {
  BufferAttribute,
  DoubleSide,
  Mesh,
  PlaneGeometry,
} from "three";
import { SurfaceGrid, sampleRow } from "../../lib/data/curveMath";
import { surfaceColor } from "../../lib/three/palette";

const WIDTH = 14;   // along threshold
const DEPTH = 7;    // across models
const HEIGHT = 3.4; // F1 = 1.0 maps to this many world units

export function useSurfaceGeometry(grid: SurfaceGrid | null) {
  return useMemo(() => {
    if (!grid || grid.rows === 0) return null;

    // One quad per (model span x threshold step). Segments are capped so a large grid does
    // not explode vertex count.
    const segX = Math.min(200, grid.cols - 1);
    const segZ = Math.max(1, (grid.rows - 1) * 6); // interpolate between model rows
    const geometry = new PlaneGeometry(WIDTH, DEPTH, segX, segZ);
    geometry.rotateX(-Math.PI / 2); // lie flat: y becomes height

    const pos = geometry.attributes.position as BufferAttribute;
    const colors = new Float32Array(pos.count * 3);

    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);

      const tx = (x + WIDTH / 2) / WIDTH;            // 0..1 across thresholds
      const tz = (z + DEPTH / 2) / DEPTH;            // 0..1 across models
      const rowF = tz * (grid.rows - 1);
      const r0 = Math.floor(rowF);
      const r1 = Math.min(grid.rows - 1, r0 + 1);
      const blend = rowF - r0;

      const f1 = sampleRow(grid.f1[r0], tx) * (1 - blend) + sampleRow(grid.f1[r1], tx) * blend;
      const prec =
        sampleRow(grid.precision[r0], tx) * (1 - blend) +
        sampleRow(grid.precision[r1], tx) * blend;

      pos.setY(i, f1 * HEIGHT);

      const c = surfaceColor(prec);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    geometry.setAttribute("color", new BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    return geometry;
  }, [grid]);
}

export function SurfaceMesh({
  grid,
  wireframe = true,
  animate = true,
}: {
  grid: SurfaceGrid | null;
  wireframe?: boolean;
  animate?: boolean;
}) {
  const geometry = useSurfaceGeometry(grid);
  const groupRef = useRef<Mesh>(null);
  const growth = useRef(0);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    if (animate && growth.current < 1) {
      growth.current = Math.min(1, growth.current + delta * 0.8);
      const e = 1 - Math.pow(1 - growth.current, 3); // ease-out cubic rise on mount
      groupRef.current.scale.y = e;
    } else if (!animate) {
      groupRef.current.scale.y = 1;
    }
  });

  if (!geometry) return null;

  return (
    <group>
      <mesh ref={groupRef} geometry={geometry} castShadow={false}>
        <meshStandardMaterial
          vertexColors
          side={DoubleSide}
          flatShading
          metalness={0.15}
          roughness={0.62}
          emissiveIntensity={0.25}
        />
      </mesh>
      {wireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial wireframe color="#8fb6ea" transparent opacity={0.12} />
        </mesh>
      )}
    </group>
  );
}

export { WIDTH as SURFACE_WIDTH, DEPTH as SURFACE_DEPTH, HEIGHT as SURFACE_HEIGHT };
