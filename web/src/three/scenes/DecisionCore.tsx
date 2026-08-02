// Chapter 3: the gradient-boosted ensemble, as a machine.
//
// The previous build drew three literal decision trees, which was honest about the shape of
// one estimator and useless as a picture of fifteen hundred of them. This is the assembly
// line instead: the feature cluster from chapter 2 arrives down four converging lanes, is
// stepped through three stages of gate, and lands in a compute core that is a projected
// tesseract -- because the thing inside it is a sum over a space with far more axes than a
// screen has. Around it, tree glyphs fire one after another in a ring, each adding its leaf
// weight to a running total that is displayed as it accumulates.
//
// What is real: the lanes converge from up-corridor because that is where the previous
// chapter physically was; three gates for the three things a boosted tree does to a row
// (route it by a split, collect a leaf weight, squash the running total); the split on the
// first gate is one the model actually learned; and the number the accumulator converges on
// is the real logit of a real held-out transaction, taken from the same sample pool the
// verdict chapter routes.
//
// One thing is illustrated rather than measured, and it is worth being explicit about: the
// *path* the running sum takes between zero and that logit. The API exposes final
// probabilities, not per-tree leaf weights, so the trajectory is drawn the way boosting
// actually behaves -- most of the distance covered by the earliest trees, the rest arriving
// as a decaying correction -- rather than replayed from a log. The endpoints are the model's;
// the curve between them is a description of the algorithm.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CylinderGeometry,
  Float32BufferAttribute,
  Group,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  Quaternion,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from "three";
import { CHAMPION } from "../../lib/data/champion";
import { sampleRisk } from "../../lib/three/hudState";
import { COLOR, HEX } from "../../lib/three/palette";
import { Label } from "../objects/Label";
import { Readout, ReadoutHandle } from "../objects/Readout";

/** Where the inbound lanes begin, up-corridor, at the mouth the feature cluster is drawn into. */
const ENTRY_Z = 15;
/** Fractions along a lane at which the three stages of gate sit. */
const STAGES = [0.62, 0.78, 0.9];
const GATE_RADIUS = [0.7, 0.54, 0.4];

/** Where the four lanes enter from, in the plane of the corridor mouth. */
const LANE_ANGLES = [0.55, 2.12, 3.69, 5.26];
const LANE_SPREAD = 3.1;

/** Number of tree glyphs standing in for the ensemble, and the ring they stand on. It sits
 *  in front of the core rather than inside it, so the estimators read as a ring around the
 *  machine instead of as wire tangled up in the tesseract. */
const TREES = 24;
const TREE_Z = 2.6;
const TREE_RING = 3.2;

const AXIS = new Vector3(0, 0, 1);
const UP = new Vector3(0, 1, 0);

interface Lane {
  start: Vector3;
  direction: Vector3;
  length: number;
  /** Turns a Z-facing ring onto the lane, so a gate is square to the flow through it. */
  gateQuaternion: Quaternion;
  /** Turns a Y-facing cylinder onto the lane, so the beam lies along it. */
  beamQuaternion: Quaternion;
}

/** The four inbound lanes, each aimed at the origin from the corridor behind it. */
function buildLanes(): Lane[] {
  return LANE_ANGLES.map((angle, i) => {
    const start = new Vector3(
      Math.cos(angle) * LANE_SPREAD,
      Math.sin(angle) * LANE_SPREAD * 0.62,
      // Staggered depth keeps the four from arriving in lockstep and reading as one cone.
      ENTRY_Z - (i % 2) * 2.6,
    );
    const direction = start.clone().negate().normalize();
    return {
      start,
      direction,
      length: start.length(),
      gateQuaternion: new Quaternion().setFromUnitVectors(AXIS, direction),
      beamQuaternion: new Quaternion().setFromUnitVectors(UP, direction),
    };
  });
}

/**
 * A tesseract, drawn as a cube inside a cube with corresponding corners joined.
 *
 * It is the standard projection of a 4-cube into three dimensions, and it is here because
 * it is the one familiar object that says "more axes than you can see" without being a
 * random polyhedron. Twenty-nine features is not four, but the point survives the licence.
 */
function tesseractGeometry(outer: number, inner: number): BufferGeometry {
  const corners = (s: number) =>
    [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => new Vector3(x * s, y * s, z * s))));

  const a = corners(outer);
  const b = corners(inner);

  const positions: number[] = [];
  const colors: number[] = [];

  const push = (p: Vector3, q: Vector3, c: [number, number, number]) => {
    positions.push(p.x, p.y, p.z, q.x, q.y, q.z);
    colors.push(...c, ...c);
  };

  // Cube edges join corners that differ in exactly one coordinate, which in this ordering
  // means indices differing by exactly one bit.
  for (const [set, tint] of [
    [a, [0.22, 0.53, 0.9] as [number, number, number]],
    [b, [0.85, 0.68, 0.33] as [number, number, number]],
  ] as const) {
    for (let i = 0; i < 8; i++) {
      for (const bit of [1, 2, 4]) {
        const j = i ^ bit;
        if (j > i) push(set[i], set[j], tint);
      }
    }
  }

  for (let i = 0; i < 8; i++) push(a[i], b[i], [0.4, 0.46, 0.62]);

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  return geometry;
}

/** One estimator, as a depth-two binary tree: a root, two splits, four leaves. */
const TREE_EDGES: Array<[number, number, number, number]> = [
  [0, 0.42, -0.19, 0.1],
  [0, 0.42, 0.19, 0.1],
  [-0.19, 0.1, -0.29, -0.24],
  [-0.19, 0.1, -0.08, -0.24],
  [0.19, 0.1, 0.08, -0.24],
  [0.19, 0.1, 0.29, -0.24],
];

/**
 * The ensemble as a ring of glyphs, all in one geometry.
 *
 * Twenty-four rather than fifteen hundred: past about thirty the ring stops reading as a
 * sequence of individual estimators and becomes a texture, which loses the one thing the
 * picture is for. The true count is on the badge below it.
 */
function buildTreeRing(): { geometry: BufferGeometry; anchors: Vector3[] } {
  const positions: number[] = [];
  const anchors: Vector3[] = [];

  for (let k = 0; k < TREES; k++) {
    const a = (k / TREES) * Math.PI * 2;
    const cx = Math.cos(a) * TREE_RING;
    const cy = Math.sin(a) * TREE_RING * 0.82;
    anchors.push(new Vector3(cx, cy, TREE_Z));

    for (const [x1, y1, x2, y2] of TREE_EDGES) {
      positions.push(cx + x1, cy + y1, TREE_Z, cx + x2, cy + y2, TREE_Z);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(new Float32Array(positions.length), 3));
  return { geometry, anchors };
}

/** Vertices per glyph in the ring geometry: six edges, two ends each. */
const VERTS_PER_TREE = TREE_EDGES.length * 2;

export function DecisionCore() {
  const lanes = useMemo(buildLanes, []);
  const ring = useMemo(buildTreeRing, []);

  const frame = useRef<LineSegments>(null);
  const cage = useRef<Group>(null);
  const core = useRef<Mesh>(null);
  const latticeRef = useRef<InstancedMesh>(null);
  const particleRef = useRef<InstancedMesh>(null);
  const leafRef = useRef<InstancedMesh>(null);
  const gates = useRef<Mesh[]>([]);
  const sumOut = useRef<ReadoutHandle>(null);
  const probOut = useRef<ReadoutHandle>(null);

  const dummy = useMemo(() => new Object3D(), []);
  const scratch = useMemo(() => new Vector3(), []);

  /* ---------------------------------------------------------------- geometry */

  const wire = useMemo(() => tesseractGeometry(2.35, 1.28), []);
  const wireMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.72,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const ringMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // A 4x4x4 block inside the inner cube: the core has structure rather than being a blob.
  const cells = useMemo(() => {
    const out: Vector3[] = [];
    for (let x = 0; x < 4; x++)
      for (let y = 0; y < 4; y++)
        for (let z = 0; z < 4; z++) out.push(new Vector3((x - 1.5) * 0.5, (y - 1.5) * 0.5, (z - 1.5) * 0.5));
    return out;
  }, []);

  const latticeGeometry = useMemo(() => new BoxGeometry(1, 1, 1), []);
  const latticeMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: HEX.brandBright,
        transparent: true,
        opacity: 0.55,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const particleGeometry = useMemo(() => new SphereGeometry(1, 8, 8), []);
  const particleMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: "#cfe4ff",
        transparent: true,
        opacity: 0.95,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const leafMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: HEX.gold,
        transparent: true,
        opacity: 0.95,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const gateGeometries = useMemo(() => GATE_RADIUS.map((r) => new TorusGeometry(r, 0.017, 8, 72)), []);

  // The lane itself, drawn. Without it the gates read as unrelated rings floating in space
  // rather than as three stages a stream passes through.
  const laneGeometries = useMemo(
    () => lanes.map((lane) => new CylinderGeometry(0.011, 0.011, lane.length, 6, 1, true)),
    [lanes],
  );
  const laneMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: "#4f7fc0",
        transparent: true,
        opacity: 0.32,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // The outbound conduit, heading down-corridor to meet the verdict gate's intake. It is the
  // only geometry in this chapter that exists for the sake of the next one.
  const outboundGeometry = useMemo(() => new CylinderGeometry(0.05, 0.09, 11, 10, 1, true), []);
  const outboundMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: HEX.brandBright,
        transparent: true,
        opacity: 0.2,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // Twenty-four packets per lane, evenly phased, so each lane reads as a steady flow rather
  // than a burst.
  const particles = useMemo(
    () =>
      Array.from({ length: lanes.length * 24 }, (_, i) => ({
        lane: i % lanes.length,
        t: (i * 0.0417) % 1,
        speed: 0.2 + ((i * 13) % 7) * 0.022,
      })),
    [lanes.length],
  );

  useEffect(
    () => () => {
      wire.dispose();
      wireMaterial.dispose();
      ring.geometry.dispose();
      ringMaterial.dispose();
      latticeGeometry.dispose();
      latticeMaterial.dispose();
      particleGeometry.dispose();
      particleMaterial.dispose();
      leafMaterial.dispose();
      gateGeometries.forEach((g) => g.dispose());
      laneGeometries.forEach((g) => g.dispose());
      laneMaterial.dispose();
      outboundGeometry.dispose();
      outboundMaterial.dispose();
    },
    [
      wire,
      wireMaterial,
      ring,
      ringMaterial,
      latticeGeometry,
      latticeMaterial,
      particleGeometry,
      particleMaterial,
      leafMaterial,
      gateGeometries,
      laneGeometries,
      laneMaterial,
      outboundGeometry,
      outboundMaterial,
    ],
  );

  /* ------------------------------------------------------------------ frames */

  /** The transaction currently being scored, and how far through the ensemble it is. */
  const pass = useRef({ logit: 1.4, index: 0 });

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;

    if (frame.current) {
      frame.current.rotation.y = t * 0.18;
      frame.current.rotation.x = Math.sin(t * 0.24) * 0.28;
    }
    if (cage.current) {
      cage.current.rotation.y = -t * 0.1;
      cage.current.rotation.z = t * 0.07;
    }
    if (core.current) {
      core.current.rotation.y = t * 0.4;
      core.current.scale.setScalar(1 + Math.sin(t * 1.8) * 0.06);
    }

    const lattice = latticeRef.current;
    if (lattice) {
      cells.forEach((cell, i) => {
        // A wave travelling out from the centre: the core is computing, not idling. The
        // cells stay axis-aligned -- tumbling them turns a lattice into loose debris.
        const wave = Math.sin(t * 2.4 - cell.length() * 2.6);
        dummy.position.copy(cell);
        dummy.scale.setScalar(0.05 + Math.max(0, wave) * 0.045);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        lattice.setMatrixAt(i, dummy.matrix);
      });
      lattice.instanceMatrix.needsUpdate = true;
    }

    const mesh = particleRef.current;
    if (mesh) {
      particles.forEach((particle, i) => {
        particle.t = (particle.t + delta * particle.speed) % 1;
        const lane = lanes[particle.lane];
        // Eased so packets decelerate into the core instead of arriving at full speed.
        const u = particle.t * particle.t * (3 - 2 * particle.t);
        dummy.position.copy(lane.start).multiplyScalar(1 - u);
        dummy.scale.setScalar(0.021 + (1 - u) * 0.011);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
    }

    gates.current.forEach((gate, i) => {
      if (!gate) return;
      const stage = i % STAGES.length;
      (gate.material as MeshBasicMaterial).userData.baseOpacity =
        0.42 + Math.max(0, Math.sin(t * 2.2 - stage * 0.9)) * 0.5;
      gate.rotation.z = t * (0.3 + stage * 0.16) * (i % 2 ? -1 : 1);
    });

    /* ----------------------------------------------- the ensemble, firing */

    // One sweep of the ring is one transaction passing through the ensemble. At the end of
    // a sweep the next real held-out probability is drawn from the pool.
    const SWEEP = 5.2;
    const cycle = (t % SWEEP) / SWEEP;
    const head = cycle * TREES;
    const index = Math.floor(head);

    if (index < pass.current.index) {
      const pool = sampleRisk.pool;
      const p = pool.length
        ? pool[Math.floor(Math.random() * pool.length)].p
        : // Before /samples answers, the champion's own tuned operating point, which is a
          // real number from a real run rather than an invented one.
          CHAMPION.threshold;
      const clamped = Math.min(0.999_9, Math.max(0.000_1, p));
      pass.current.logit = Math.log(clamped / (1 - clamped));
    }
    pass.current.index = index;

    // Boosting covers most of the distance in its earliest trees and spends the rest
    // correcting, so the accumulator approaches its final value along a decaying exponential
    // with a small oscillation rather than climbing in a straight line.
    const settle = 1 - Math.exp(-3.4 * cycle);
    const wobble = Math.exp(-4.5 * cycle) * Math.sin(cycle * 26) * 0.42;
    const running = pass.current.logit * settle + wobble;
    const probability = 1 / (1 + Math.exp(-running));

    sumOut.current?.set(`${running >= 0 ? "+" : ""}${running.toFixed(2)}`);
    probOut.current?.set(probability.toFixed(3));

    // The glyphs light in sequence with a decaying tail behind the head, so the ring reads
    // as one estimator firing at a time rather than as a ring that pulses.
    const colours = ring.geometry.getAttribute("color") as BufferAttribute;
    const array = colours.array as Float32Array;
    for (let k = 0; k < TREES; k++) {
      const behind = (head - k + TREES) % TREES;
      const glow = Math.exp(-behind * 0.62);
      const c = glow > 0.5 ? COLOR.gold : COLOR.brand;
      const level = 0.085 + glow * 1.75;
      for (let v = 0; v < VERTS_PER_TREE; v++) {
        const o = (k * VERTS_PER_TREE + v) * 3;
        array[o] = c.r * level;
        array[o + 1] = c.g * level;
        array[o + 2] = c.b * level;
      }
    }
    colours.needsUpdate = true;

    // Each glyph throws its leaf weight at the core as it fires.
    const leaves = leafRef.current;
    if (leaves) {
      for (let k = 0; k < TREES; k++) {
        const behind = (head - k + TREES) % TREES;
        const travel = Math.min(1, behind / 2.6);
        scratch.copy(ring.anchors[k]).multiplyScalar(1 - travel * travel);
        dummy.position.copy(scratch);
        dummy.scale.setScalar(behind < 2.6 ? 0.055 * (1 - travel) + 0.012 : 0);
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        leaves.setMatrixAt(k, dummy.matrix);
      }
      leaves.instanceMatrix.needsUpdate = true;
    }
  });

  /* ------------------------------------------------------------------ render */

  return (
    <group>
      {/* The core. */}
      <lineSegments ref={frame} geometry={wire} material={wireMaterial} frustumCulled={false} />

      <group ref={cage}>
        <instancedMesh
          ref={latticeRef}
          args={[latticeGeometry, latticeMaterial, cells.length]}
          frustumCulled={false}
        />
      </group>

      <mesh ref={core}>
        <icosahedronGeometry args={[0.44, 2]} />
        <meshBasicMaterial
          color={HEX.gold}
          transparent
          opacity={0.9}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* The ensemble, firing one estimator at a time. */}
      <lineSegments geometry={ring.geometry} material={ringMaterial} frustumCulled={false} />
      <instancedMesh ref={leafRef} args={[particleGeometry, leafMaterial, TREES]} frustumCulled={false} />

      {/* The four inbound lanes, arriving from where chapter 2 was. */}
      {lanes.map((lane, s) => (
        <mesh
          key={`lane-${s}`}
          geometry={laneGeometries[s]}
          material={laneMaterial}
          position={lane.start.clone().multiplyScalar(0.5)}
          quaternion={lane.beamQuaternion}
        />
      ))}

      {/* Three stages of gate on each of them. */}
      {lanes.flatMap((lane, s) =>
        STAGES.map((fraction, stage) => {
          const index = s * STAGES.length + stage;
          return (
            <mesh
              key={index}
              ref={(m) => {
                if (m) gates.current[index] = m;
              }}
              geometry={gateGeometries[stage]}
              position={lane.start.clone().multiplyScalar(1 - fraction)}
              quaternion={lane.gateQuaternion}
            >
              <meshBasicMaterial
                color={stage === STAGES.length - 1 ? HEX.gold : HEX.brand}
                transparent
                opacity={0.6}
                blending={AdditiveBlending}
                depthWrite={false}
              />
            </mesh>
          );
        }),
      )}

      <instancedMesh
        ref={particleRef}
        args={[particleGeometry, particleMaterial, particles.length]}
        frustumCulled={false}
      />

      {/* Out the far side, toward the gate that has to decide what the number means. */}
      <mesh
        geometry={outboundGeometry}
        material={outboundMaterial}
        position={[0, 0, -8]}
        rotation={[Math.PI / 2, 0, 0]}
      />

      {/* Live, and the reason the chapter is worth watching rather than reading: the running
          sum of leaf weights, and what the logistic gate makes of it. */}
      <Readout
        ref={sumOut}
        caption="SUM OF LEAF WEIGHTS"
        initial="+0.00"
        position={[-2.72, -2.95, 2.6]}
        height={0.52}
        color={HEX.brandBright}
      />
      <Readout
        ref={probOut}
        caption="P(FRAUD)"
        initial="0.500"
        position={[2.72, -2.95, 2.6]}
        height={0.52}
        color={HEX.gold}
      />

      {/* One name per stage, and each on a different lane. Every lane has all three stages, so
          which one carries which label is arbitrary -- but putting all three on the same lane
          stacks them into a pile at the middle of frame, because consecutive stages differ
          mostly in depth and a sprite's screen position barely moves. Spreading them around
          the funnel separates them without saying anything untrue. */}
      {/* Kept to ASCII on purpose: these are rasterised in the monospace stack, and a sigma
          silently falls back to a different face and comes out as the wrong glyph. */}
      {(
        [
          ["V14 < -2.31", 0, 1],
          ["leaf weight", 1, 3],
          ["logistic gate", 2, 0],
        ] as const
      ).map(([text, stage, lane]) => {
        const anchor = lanes[lane].start;
        const out = 1 - STAGES[stage];
        return (
          <Label
            key={text}
            text={text}
            pill
            size={64}
            position={[
              anchor.x * out,
              anchor.y * out + Math.sign(anchor.y || 1) * (GATE_RADIUS[stage] + 0.44),
              anchor.z * out,
            ]}
            height={0.26}
            color={stage === 2 ? HEX.gold : "#9fb4cc"}
          />
        );
      })}

      <Label
        text="1500 trees · 15 leaves · depth-wise"
        pill
        size={64}
        position={[0, -4.05, 2.6]}
        height={0.3}
        color={HEX.brandBright}
      />
    </group>
  );
}
