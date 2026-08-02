// Chapter 5: the console.
//
// The camera comes out the far side of the routing hub and settles in front of the whole
// engine at rest. What it is looking at is the confusion matrix, built as four pillars whose
// heights are the real counts at whatever threshold was left on the slider one chapter ago.
// Drag the threshold anywhere in the story and these four pillars move, because they are the
// same sweep of the same test set that every other number on the page is quoting.
//
// The heights are logarithmic and the scene says so on the caption. It is not a stylistic
// choice: the true negatives outnumber the true positives eight hundred to one, and on a
// linear axis three of the four pillars are invisible. A log axis is the only way this
// particular matrix can be drawn at all, and hiding that fact would be the dishonest option.
//
// The inject button in the rail fires a synthetic high-risk payment down the corridor at the
// matrix. Whether it lands in the caught pillar or the missed one is decided by the threshold
// the reader has chosen, which makes it the shortest possible statement of the whole page.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BoxGeometry,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  InstancedMesh,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  SphereGeometry,
  Vector3,
} from "three";
import { pointAt } from "../../lib/data/champion";
import { curve, hud, sampleRisk } from "../../lib/three/hudState";
import { COLOR, HEX } from "../../lib/three/palette";
import { Label } from "../objects/Label";
import { Readout, ReadoutHandle } from "../objects/Readout";

/** Probability assigned to the injected test payment. Deliberately high, and deliberately
 *  not 1.0 -- a threshold dragged to the top should still be able to miss it. */
const ANOMALY_RISK = 0.97;

/** Where an injected payment enters from, up-corridor. */
const INJECT_FROM = new Vector3(0, 3.4, 26);

/** Half the spacing of the matrix, in x and z. */
const GAP_X = 1.72;
const GAP_Z = 1.15;

type CellId = "TN" | "FP" | "FN" | "TP";

interface Cell {
  id: CellId;
  caption: string;
  x: number;
  z: number;
  color: string;
  tint: Color;
}

/**
 * The matrix, laid out the way it is written down: actual class on one axis, predicted class
 * on the other, so the two error cells sit off the diagonal where they belong.
 */
const CELLS: Cell[] = [
  { id: "TN", caption: "TRUE NEGATIVE", x: -GAP_X, z: -GAP_Z, color: HEX.brand, tint: COLOR.brand },
  { id: "FP", caption: "FALSE ALARM", x: GAP_X, z: -GAP_Z, color: HEX.warn, tint: COLOR.warn },
  { id: "FN", caption: "MISSED FRAUD", x: -GAP_X, z: GAP_Z, color: HEX.fraud, tint: COLOR.fraud },
  { id: "TP", caption: "CAUGHT", x: GAP_X, z: GAP_Z, color: HEX.legit, tint: COLOR.legit },
];

/** Log height, so a pillar of 56,649 and a pillar of 2 can share an axis. */
function pillarHeight(count: number): number {
  return Math.log10(count + 1) * 0.74;
}

/** The floor the matrix stands on: two axes and a light grid. */
function floorGeometry(): BufferGeometry {
  const positions: number[] = [];
  const colors: number[] = [];
  const push = (a: Vector3, b: Vector3, level: number) => {
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    for (let i = 0; i < 2; i++) colors.push(0.22 * level, 0.35 * level, 0.56 * level);
  };

  const half = 2.85;
  for (let i = -6; i <= 6; i++) {
    const u = (i / 6) * half;
    const edge = Math.abs(i) === 6 || i === 0;
    push(new Vector3(-half, 0, u), new Vector3(half, 0, u), edge ? 2.4 : 0.85);
    push(new Vector3(u, 0, -half), new Vector3(u, 0, half), edge ? 2.4 : 0.85);
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  return geometry;
}

const AMBIENT = 150;

export function ConsoleScene() {
  const pillars = useRef<Mesh[]>([]);
  const counts = useRef<Array<ReadoutHandle | null>>([]);
  const ambientRef = useRef<InstancedMesh>(null);
  const shotRef = useRef<Mesh>(null);
  const waveRef = useRef<Mesh>(null);
  const verdictOut = useRef<ReadoutHandle>(null);

  const dummy = useMemo(() => new Object3D(), []);
  const scratch = useMemo(() => new Vector3(), []);
  const colour = useMemo(() => new Color(), []);

  /* ---------------------------------------------------------------- geometry */

  const floor = useMemo(floorGeometry, []);
  const floorMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.5,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // Anchored at its base rather than its centre, so setting scale.y grows the pillar upward
  // out of the floor instead of through it.
  const pillarGeometry = useMemo(() => {
    const g = new BoxGeometry(0.95, 1, 0.95);
    g.translate(0, 0.5, 0);
    return g;
  }, []);

  const pillarMaterials = useMemo(
    () =>
      CELLS.map(
        (cell) =>
          new MeshBasicMaterial({
            color: cell.color,
            transparent: true,
            opacity: 0.34,
            blending: AdditiveBlending,
            depthWrite: false,
            side: DoubleSide,
          }),
      ),
    [],
  );

  const dotGeometry = useMemo(() => new SphereGeometry(1, 8, 8), []);
  const ambientMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        transparent: true,
        opacity: 0.75,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // The engine still running behind the summary: real held-out transactions orbiting on a
  // wide, slow ellipse, coloured by the verdict the current threshold gives them.
  const ambient = useMemo(
    () =>
      Array.from({ length: AMBIENT }, (_, i) => ({
        angle: (i / AMBIENT) * Math.PI * 2,
        radius: 6.4 + ((i * 13) % 11) * 0.31,
        y: -1.6 + ((i * 7) % 17) * 0.29,
        speed: 0.055 + ((i * 5) % 6) * 0.011,
        risk: i % 6 === 0 ? 0.8 : 0.03,
      })),
    [],
  );

  useEffect(
    () => () => {
      floor.dispose();
      floorMaterial.dispose();
      pillarGeometry.dispose();
      pillarMaterials.forEach((m) => m.dispose());
      dotGeometry.dispose();
      ambientMaterial.dispose();
    },
    [floor, floorMaterial, pillarGeometry, pillarMaterials, dotGeometry, ambientMaterial],
  );

  /* ------------------------------------------------------------------ frames */

  /** The injected payment: which click it came from, how far along it is, and its fate. */
  const shot = useRef({ seen: 0, t: 1, caught: false });

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const threshold = hud.threshold;
    const point = pointAt(curve.points, threshold);
    const value: Record<CellId, number> = {
      TN: point.TN,
      FP: point.FP,
      FN: point.FN,
      TP: point.TP,
    };

    CELLS.forEach((cell, i) => {
      const pillar = pillars.current[i];
      if (pillar) {
        // Eased toward the target rather than assigned, so dragging the threshold makes the
        // matrix move like a physical readout instead of snapping between states.
        const target = pillarHeight(value[cell.id]);
        pillar.scale.y += (target - pillar.scale.y) * Math.min(1, delta * 7);
        // The two error cells breathe; the two correct ones sit still. It is the cheapest
        // way to make the eye land on the numbers that cost money.
        const error = cell.id === "FP" || cell.id === "FN";
        pillarMaterials[i].userData.baseOpacity =
          0.3 + (error ? Math.max(0, Math.sin(t * 2.4 - i)) * 0.34 : 0.1);
      }
      counts.current[i]?.set(value[cell.id].toLocaleString());
    });

    /* --------------------------------------------------------- ambient stream */

    const orbit = ambientRef.current;
    if (orbit) {
      ambient.forEach((dot, i) => {
        dot.angle += delta * dot.speed;
        scratch.set(
          Math.cos(dot.angle) * dot.radius,
          dot.y + Math.sin(t * 0.3 + i) * 0.12,
          Math.sin(dot.angle) * dot.radius * 0.55,
        );
        dummy.position.copy(scratch);
        dummy.scale.setScalar(0.032);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        orbit.setMatrixAt(i, dummy.matrix);

        // Sampled from the live pool once it has answered, so the ring behind the summary is
        // the same population the summary is about.
        const pool = sampleRisk.pool;
        const risk = pool.length ? pool[i % pool.length].p : dot.risk;
        colour.copy(risk >= threshold ? COLOR.fraud : COLOR.legit).multiplyScalar(0.72);
        orbit.setColorAt(i, colour);
      });
      orbit.instanceMatrix.needsUpdate = true;
      if (orbit.instanceColor) orbit.instanceColor.needsUpdate = true;
    }

    /* -------------------------------------------------------------- injection */

    if (hud.inject !== shot.current.seen) {
      shot.current.seen = hud.inject;
      shot.current.t = 0;
      shot.current.caught = ANOMALY_RISK >= threshold;
      verdictOut.current?.set(shot.current.caught ? "FLAGGED" : "MISSED");
    }

    const s = shot.current;
    if (s.t < 1) s.t = Math.min(1, s.t + delta * 0.72);

    const target = CELLS.find((c) => c.id === (s.caught ? "TP" : "FN"))!;
    const flight = Math.min(1, s.t / 0.66);

    if (shotRef.current) {
      const eased = flight * flight * (3 - 2 * flight);
      shotRef.current.position.set(
        INJECT_FROM.x + (target.x - INJECT_FROM.x) * eased,
        INJECT_FROM.y + (pillarHeight(s.caught ? point.TP : point.FN) + 0.3 - INJECT_FROM.y) * eased,
        INJECT_FROM.z + (target.z - INJECT_FROM.z) * eased,
      );
      shotRef.current.scale.setScalar(s.t < 1 ? 0.1 + (1 - flight) * 0.16 : 0);
      const material = shotRef.current.material as MeshBasicMaterial;
      material.color.copy(s.caught ? COLOR.legit : COLOR.fraud);
      material.userData.baseOpacity = s.t < 1 ? 1 : 0;
    }

    if (waveRef.current) {
      // The impact, as one ring expanding out of the pillar it landed in.
      const burst = Math.max(0, (s.t - 0.62) / 0.38);
      waveRef.current.position.set(target.x, 0.04, target.z);
      waveRef.current.scale.setScalar(0.2 + burst * 5.5);
      (waveRef.current.material as MeshBasicMaterial).userData.baseOpacity =
        burst > 0 && burst < 1 ? (1 - burst) * 0.75 : 0;
      (waveRef.current.material as MeshBasicMaterial).color.copy(
        s.caught ? COLOR.legit : COLOR.fraud,
      );
    }
  });

  /* ------------------------------------------------------------------ render */

  return (
    <group position={[0, -0.9, 0]}>
      <lineSegments geometry={floor} material={floorMaterial} frustumCulled={false} />

      {CELLS.map((cell, i) => (
        <group key={cell.id} position={[cell.x, 0, cell.z]}>
          <mesh
            ref={(m) => {
              if (m) pillars.current[i] = m;
            }}
            material={pillarMaterials[i]}
            geometry={pillarGeometry}
            scale={[1, 0.05, 1]}
          />
          <Readout
            ref={(handle) => {
              counts.current[i] = handle;
            }}
            caption={cell.caption}
            initial="—"
            position={[0, -0.34, 0.66]}
            height={0.36}
            color={cell.color}
          />
        </group>
      ))}

      {/* No axis labels: each pillar already carries its own name on the plate under it, and
          a second set of captions across the front of the matrix only ever collided with
          them. The one thing the pillars cannot say for themselves is what their heights
          mean, so that is the only caption left. */}
      <Label
        text="log10 scale · counts on 56,746 held-out rows"
        pill
        size={64}
        position={[0, 4.05, -1.2]}
        height={0.3}
        color="#9fb4cc"
      />

      {/* The injected payment, and the ring it makes when it lands. */}
      <mesh ref={shotRef} scale={0}>
        <sphereGeometry args={[1, 14, 14]} />
        <meshBasicMaterial
          color={HEX.fraud}
          transparent
          opacity={0}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <mesh ref={waveRef} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.82, 1, 64]} />
        <meshBasicMaterial
          color={HEX.fraud}
          transparent
          opacity={0}
          blending={AdditiveBlending}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
      <Readout
        ref={verdictOut}
        caption="LAST INJECTED PAYMENT"
        initial="—"
        position={[0, 4.8, -1.2]}
        height={0.5}
        color={HEX.gold}
      />

      <instancedMesh
        ref={ambientRef}
        args={[dotGeometry, ambientMaterial, ambient.length]}
        frustumCulled={false}
      />
    </group>
  );
}
