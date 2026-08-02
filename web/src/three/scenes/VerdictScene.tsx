// Chapter 4: the routing hub.
//
// Scored transactions arrive down a single power conduit, hit the decision gate, and are
// thrown left into an emerald shield or right into a crimson containment vault. The gate is
// a real mechanism: an outer stator with twelve teeth, a prism inside it, and a splitter
// blade across the aperture. Moving the threshold slider in the rail turns the prism, sweeps
// the blade, reweights the gate's colour and re-routes the stream, all in the same frame.
//
// Everything here is real in three separate senses, and the chapter is worth nothing if any
// of them slips.
//
//   The probabilities are real. Each packet carries the model's actual predicted probability
//   for a held-out transaction, pulled from GET /samples.
//
//   The labels are real. Each packet also carries whether that transaction was genuinely
//   fraudulent, which is what lets the scene draw its own mistakes: a crimson packet going
//   down the emerald lane is fraud the current threshold is waving through, and it is drawn
//   as one because it is one.
//
//   The cut-off is real. It is the same number the slider writes, and the same number the
//   readouts beside it are computed from. The model is fixed; the threshold is a choice; the
//   choice is what decides outcomes, and this is the chapter that says so.
//
// The two stress states are the two ways an operating point fails, and they are opposite
// ends of the same slider. Drag it down and the vault floods with legitimate customers.
// Drag it up and fraud walks through the green lane. Both are shown; neither is invented.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  InstancedMesh,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  QuadraticBezierCurve3,
  Quaternion,
  RingGeometry,
  SphereGeometry,
  TorusGeometry,
  Vector3,
} from "three";
import { alarm, hud, routed, sampleRisk } from "../../lib/three/hudState";
import { COLOR, HEX } from "../../lib/three/palette";
import { Label } from "../objects/Label";

/** Where the conduit starts, up-corridor, overlapping the core's outbound stub. */
const CONDUIT = 30;
const GATE = new Vector3(0, 0, 0);

/** Fraction of a packet's path at which it is committed to a lane. */
const GATE_T = 0.42;
/** Fraction at which it stops travelling and starts orbiting its zone. */
const ORBIT_T = 0.72;

const ZONE = {
  legit: new Vector3(-4.7, -0.5, -7.6),
  fraud: new Vector3(4.7, 0.5, -7.6),
};
const ORBIT_RADIUS = 1.35;

const AXIS = new Vector3(0, 0, 1);
const UP = new Vector3(0, 1, 0);

interface Lane {
  centre: Vector3;
  /** In-plane basis, so a packet's orbit is a plain pair of sines. */
  u: Vector3;
  v: Vector3;
  quaternion: Quaternion;
  curve: QuadraticBezierCurve3;
}

function buildLane(centre: Vector3, normal: Vector3): Lane {
  const n = normal.clone().normalize();
  const u = UP.clone().cross(n).normalize();
  const v = n.clone().cross(u).normalize();
  const entry = centre.clone().addScaledVector(u, ORBIT_RADIUS);
  // The control point sits on the corridor axis, so a packet leaves the gate travelling
  // straight and is *deflected* rather than turning the moment it is told which way to go.
  const control = new Vector3(centre.x * 0.22, centre.y * 0.22, -3.1);

  return {
    centre,
    u,
    v,
    quaternion: new Quaternion().setFromUnitVectors(AXIS, n),
    curve: new QuadraticBezierCurve3(GATE.clone(), control, entry),
  };
}

/* ---------------------------------------------------------------- the gate */

/**
 * The decision gate: a stator with twelve teeth, a prism inside it, and a blade across the
 * aperture.
 *
 * The prism's angle is the threshold. That is the whole point of building it as a mechanism
 * rather than as a glowing ring -- a number between nought and one is abstract, and a thing
 * that visibly turns when you drag a slider is not. The rotation is damped rather than
 * bound directly, so the gate reads as machinery being driven rather than as a value being
 * assigned.
 */
function DecisionGate() {
  const prism = useRef<Group>(null);
  const stator = useRef<Group>(null);
  const blade = useRef<Mesh>(null);
  const aperture = useRef<Mesh>(null);
  const angle = useRef(0);

  const tint = useMemo(() => new Color(), []);

  const teeth = useMemo(() => Array.from({ length: 12 }, (_, i) => (i / 12) * Math.PI * 2), []);

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const threshold = hud.threshold;

    // Damped, not assigned: the mechanism catches up with the slider over about a tenth of
    // a second, which is what makes it feel driven rather than teleported.
    angle.current += (threshold - angle.current) * Math.min(1, delta * 11);
    const a = angle.current;

    if (prism.current) {
      prism.current.rotation.z = -a * Math.PI * 1.15;
      prism.current.rotation.x = 0.22 + a * 0.42;
      prism.current.scale.setScalar(0.82 + a * 0.34 + (hud.scrubbing ? 0.12 : 0));
    }
    if (stator.current) {
      stator.current.rotation.z = t * (hud.scrubbing ? 0.5 : 0.14);
    }
    if (blade.current) {
      // The blade sweeps across the aperture: the higher the cut-off, the more of the
      // opening is closed to the crimson side.
      blade.current.rotation.z = (a - 0.5) * 2.1;
      const material = blade.current.material as MeshBasicMaterial;
      material.userData.baseOpacity = 0.32 + (hud.scrubbing ? 0.35 : 0.12) * (0.6 + Math.sin(t * 6) * 0.4);
    }
    if (aperture.current) {
      const flare = hud.scrubbing ? 1.3 : 1;
      aperture.current.scale.setScalar((0.92 + Math.sin(t * 2) * 0.03) * flare);
      aperture.current.rotation.z = -t * 0.26;
      const material = aperture.current.material as MeshBasicMaterial;
      // Green-weighted when the gate is letting traffic through, crimson-weighted when it is
      // catching everything. The colour of the gate is the operating point.
      tint.copy(COLOR.fraud).lerp(COLOR.legit, Math.min(1, a * 1.25));
      material.color.copy(tint);
      material.userData.baseOpacity = hud.scrubbing ? 0.95 : 0.6;
    }
  });

  return (
    <group>
      {/* Stator: the fixed outer housing, and the teeth that make it read as a machined
          part rather than as a hoop. */}
      <group ref={stator}>
        <mesh>
          <torusGeometry args={[1.42, 0.045, 10, 88]} />
          <meshBasicMaterial
            color={HEX.brand}
            transparent
            opacity={0.55}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
        {teeth.map((a) => (
          <mesh key={a} position={[Math.cos(a) * 1.6, Math.sin(a) * 1.6, 0]} rotation={[0, 0, a]}>
            <boxGeometry args={[0.3, 0.035, 0.035]} />
            <meshBasicMaterial
              color={HEX.brandBright}
              transparent
              opacity={0.5}
              blending={AdditiveBlending}
              depthWrite={false}
            />
          </mesh>
        ))}
      </group>

      {/* The prism itself: an eight-sided barrel that turns with the threshold. */}
      <group ref={prism}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.94, 0.94, 0.5, 8, 1, true]} />
          <meshBasicMaterial
            color={HEX.brandBright}
            wireframe
            transparent
            opacity={0.5}
            blending={AdditiveBlending}
            depthWrite={false}
            side={DoubleSide}
          />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.62, 0.62, 0.54, 8, 1, true]} />
          <meshBasicMaterial
            color={HEX.gold}
            transparent
            opacity={0.16}
            blending={AdditiveBlending}
            depthWrite={false}
            side={DoubleSide}
          />
        </mesh>
      </group>

      {/* The blade across the opening. */}
      <mesh ref={blade} position={[0, 0, 0.02]}>
        <planeGeometry args={[2.5, 0.075]} />
        <meshBasicMaterial
          color={HEX.warn}
          transparent
          opacity={0.35}
          blending={AdditiveBlending}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>

      {/* The aperture, whose colour is the operating point. */}
      <mesh ref={aperture}>
        <ringGeometry args={[1.02, 1.13, 76]} />
        <meshBasicMaterial
          color={HEX.brand}
          transparent
          opacity={0.6}
          blending={AdditiveBlending}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------- zones */

/** Expanding wave rings: the approved side, and the only calm thing in the chapter. */
function ShieldZone({ centre }: { centre: Vector3 }) {
  const waves = useRef<Mesh[]>([]);

  const geometry = useMemo(() => new RingGeometry(0.95, 1.02, 80), []);
  // One material per wave rather than one shared: each ring fades on its own schedule, and
  // opacity lives on the material.
  const materials = useMemo(
    () =>
      [0, 1, 2].map(
        () =>
          new MeshBasicMaterial({
            color: HEX.legit,
            transparent: true,
            opacity: 0.42,
            blending: AdditiveBlending,
            depthWrite: false,
            side: DoubleSide,
          }),
      ),
    [],
  );

  useEffect(
    () => () => {
      geometry.dispose();
      materials.forEach((m) => m.dispose());
    },
    [geometry, materials],
  );

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    waves.current.forEach((wave, i) => {
      if (!wave) return;
      // Three rings out of phase on one slow cycle: a shield breathing rather than pulsing.
      const u = ((t * 0.32 + i / 3) % 1);
      wave.scale.setScalar(0.55 + u * 2.4);
      wave.rotation.z = t * 0.08 * (i % 2 ? -1 : 1);
      const material = wave.material as MeshBasicMaterial;
      material.userData.baseOpacity = Math.sin(u * Math.PI) * 0.42;
    });
  });

  return (
    <group position={centre}>
      {materials.map((material, i) => (
        <mesh
          key={i}
          ref={(m) => {
            if (m) waves.current[i] = m;
          }}
          geometry={geometry}
          material={material}
          rotation={[0.35, 0.2, 0]}
        />
      ))}
      <mesh rotation={[0.35, 0.2, 0]}>
        <torusGeometry args={[ORBIT_RADIUS, 0.014, 8, 110]} />
        <meshBasicMaterial
          color={HEX.legit}
          transparent
          opacity={0.7}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/** How many arcs the vault throws, and how many kinks each one has. */
const ARCS = 9;
const ARC_STEPS = 6;

/**
 * The flagged side: a cage, three hazard rings turning against each other, and arcs that
 * jump between the cage and whatever is inside it.
 *
 * The arcs are regenerated on a timer rather than every frame. At sixty rebuilds a second
 * they average out into a static fuzz; at eight they read as discharges.
 */
function ContainmentVault({ centre }: { centre: Vector3 }) {
  const rings = useRef<Mesh[]>([]);
  const cage = useRef<Mesh>(null);
  const coreRef = useRef<Mesh>(null);
  const flare = useRef<Mesh>(null);
  const nextArc = useRef(0);

  const ringGeometries = useMemo(
    () => [
      new TorusGeometry(1.74, 0.013, 8, 96),
      new TorusGeometry(2.0, 0.01, 8, 96),
      new TorusGeometry(2.26, 0.008, 8, 96),
    ],
    [],
  );
  const ringMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: HEX.fraud,
        transparent: true,
        opacity: 0.6,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const arcGeometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(new Float32Array(ARCS * ARC_STEPS * 6), 3));
    return g;
  }, []);
  const arcMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        color: HEX.fraud,
        transparent: true,
        opacity: 0.75,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  useEffect(
    () => () => {
      ringGeometries.forEach((g) => g.dispose());
      ringMaterial.dispose();
      arcGeometry.dispose();
      arcMaterial.dispose();
    },
    [ringGeometries, ringMaterial, arcGeometry, arcMaterial],
  );

  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const flood = alarm.flood;

    ringMaterial.userData.baseOpacity = 0.34 + Math.sin(t * 2.6) * 0.16 + flood * 0.4;
    rings.current.forEach((ring, i) => {
      if (!ring) return;
      // Each cage spins on its own axis, faster as the vault fills, so the anomaly reads as
      // contained rather than decorated -- and as straining when the threshold is wrong.
      const rate = 0.42 + i * 0.24 + flood * 1.6;
      ring.rotation.x = t * rate * (i % 2 ? -1 : 1);
      ring.rotation.y = t * (0.3 + i * 0.17 + flood);
    });

    if (cage.current) {
      cage.current.rotation.y = t * 0.22;
      cage.current.rotation.x = -t * 0.15;
    }
    if (coreRef.current) {
      coreRef.current.scale.setScalar(0.34 + Math.sin(t * 4) * 0.05 + flood * 0.3);
    }
    if (flare.current) {
      // The alarm flare. Dark at a sane operating point, and impossible to miss when the
      // vault is taking legitimate customers.
      const beat = 0.5 + Math.sin(t * 9) * 0.5;
      flare.current.scale.setScalar(1.6 + flood * 1.4);
      (flare.current.material as MeshBasicMaterial).userData.baseOpacity = flood * flood * beat * 0.34;
    }

    if (t >= nextArc.current) {
      nextArc.current = t + 0.1 + Math.random() * 0.06;
      const attribute = arcGeometry.getAttribute("position") as BufferAttribute;
      const array = attribute.array as Float32Array;
      for (let a = 0; a < ARCS; a++) {
        // From a point on the core out to a point on the cage, kinked on the way.
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const dir = new Vector3(
          Math.sin(phi) * Math.cos(theta),
          Math.sin(phi) * Math.sin(theta),
          Math.cos(phi),
        );
        for (let s = 0; s < ARC_STEPS; s++) {
          const r0 = 0.4 + (s / ARC_STEPS) * 1.34;
          const r1 = 0.4 + ((s + 1) / ARC_STEPS) * 1.34;
          const jitter = 0.16;
          const o = (a * ARC_STEPS + s) * 6;
          array[o + 0] = dir.x * r0 + (Math.random() - 0.5) * jitter;
          array[o + 1] = dir.y * r0 + (Math.random() - 0.5) * jitter;
          array[o + 2] = dir.z * r0 + (Math.random() - 0.5) * jitter;
          array[o + 3] = dir.x * r1 + (Math.random() - 0.5) * jitter;
          array[o + 4] = dir.y * r1 + (Math.random() - 0.5) * jitter;
          array[o + 5] = dir.z * r1 + (Math.random() - 0.5) * jitter;
        }
      }
      attribute.needsUpdate = true;
    }
    arcMaterial.userData.baseOpacity = 0.24 + flood * 0.55;
  });

  return (
    <group position={centre}>
      <mesh ref={cage}>
        <octahedronGeometry args={[1.5, 1]} />
        <meshBasicMaterial
          color={HEX.fraud}
          wireframe
          transparent
          opacity={0.22}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {ringGeometries.map((geometry, i) => (
        <mesh
          key={i}
          ref={(m) => {
            if (m) rings.current[i] = m;
          }}
          geometry={geometry}
          material={ringMaterial}
        />
      ))}

      <lineSegments geometry={arcGeometry} material={arcMaterial} frustumCulled={false} />

      <mesh ref={coreRef}>
        <sphereGeometry args={[1, 16, 16]} />
        <meshBasicMaterial
          color={HEX.fraud}
          transparent
          opacity={0.85}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      <mesh ref={flare}>
        <sphereGeometry args={[1, 20, 20]} />
        <meshBasicMaterial
          color="#ff7a6e"
          transparent
          opacity={0}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      <mesh rotation={[0.35, -0.2, 0]}>
        <torusGeometry args={[ORBIT_RADIUS, 0.016, 8, 110]} />
        <meshBasicMaterial
          color={HEX.fraud}
          transparent
          opacity={0.8}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------- scene */

interface Packet {
  t: number;
  speed: number;
  /** Model probability for a real held-out transaction. */
  risk: number;
  /** Whether that transaction was genuinely fraudulent. */
  truth: boolean;
  /** Which way the gate sent it. */
  flagged: boolean;
  committed: boolean;
  /** Lateral offset inside the conduit, so the inbound stream has width. */
  angle: number;
  spread: number;
}

const COUNT = 96;

export function VerdictScene() {
  const lanes = useMemo(
    () => ({
      legit: buildLane(ZONE.legit, new Vector3(0.32, 0.4, 0.86)),
      fraud: buildLane(ZONE.fraud, new Vector3(-0.32, -0.36, 0.88)),
    }),
    [],
  );

  const packetRef = useRef<InstancedMesh>(null);

  const dummy = useMemo(() => new Object3D(), []);
  const scratch = useMemo(() => new Vector3(), []);
  const colour = useMemo(() => new Color(), []);

  /* ---------------------------------------------------------------- geometry */

  const packetGeometry = useMemo(() => new SphereGeometry(1, 10, 10), []);
  const packetMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        transparent: true,
        opacity: 1,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // The conduit: an open tube for the stream itself, plus support rings along it. Both are
  // what make the approach read as engineered rather than as a beam drawn in space.
  const tubeGeometry = useMemo(() => new CylinderGeometry(0.42, 0.16, CONDUIT, 12, 1, true), []);
  const supportGeometry = useMemo(() => new TorusGeometry(0.36, 0.012, 6, 40), []);
  // Ribs along the conduit, starting well up-corridor. They exist to be flown past on the
  // approach; any closer to the gate and the nearest one fills the frame during the hold,
  // which is a hoop in front of the lens rather than a piece of the machine.
  const supports = useMemo(
    () => Array.from({ length: 7 }, (_, i) => 12 + (i / 6) * (CONDUIT - 13.5)),
    [],
  );
  const laneBeams = useMemo(
    () => ({
      legit: new TubeGeometryLite(lanes.legit.curve),
      fraud: new TubeGeometryLite(lanes.fraud.curve),
    }),
    [lanes],
  );

  useEffect(
    () => () => {
      packetGeometry.dispose();
      packetMaterial.dispose();
      tubeGeometry.dispose();
      supportGeometry.dispose();
      laneBeams.legit.dispose();
      laneBeams.fraud.dispose();
    },
    [packetGeometry, packetMaterial, tubeGeometry, supportGeometry, laneBeams],
  );

  const packets = useMemo<Packet[]>(
    () =>
      Array.from({ length: COUNT }, (_, i) => ({
        t: (i / COUNT + (i % 7) * 0.013) % 1,
        speed: 0.15 + ((i * 11) % 6) * 0.016,
        risk: i % 7 === 0 ? 0.9 : 0.02,
        truth: i % 7 === 0,
        flagged: false,
        committed: false,
        angle: (i * 2.39996323) % (Math.PI * 2),
        spread: 0.06 + ((i * 5) % 9) * 0.036,
      })),
    [],
  );

  /**
   * Rolling counts of what the gate has done lately, as exponentially decayed sums.
   *
   * A plain running total would take a minute to respond to the slider, which defeats the
   * point; a short window would flicker. Decaying by a fixed factor on every commit gives a
   * readout that settles in a couple of seconds and holds still once it has.
   */
  const window = useRef({ legit: 0, legitFlagged: 0, fraud: 0, fraudMissed: 0 });

  const nextSample = (i: number): { p: number; fraud: boolean } => {
    const pool = sampleRisk.pool;
    if (pool.length === 0) {
      // Before /samples answers, a deterministic mix: roughly one in six shown as risky,
      // since a literal 0.167 percent would mean an empty crimson lane for minutes.
      const risky = i % 6 === 0;
      return { p: risky ? 0.72 + (i % 3) * 0.09 : 0.01 + (i % 5) * 0.02, fraud: risky };
    }
    return pool[(i * 7 + Math.floor(Math.random() * pool.length)) % pool.length];
  };

  /* ------------------------------------------------------------------ frames */

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const threshold = hud.threshold;
    const mesh = packetRef.current;
    if (!mesh) return;

    const w = window.current;

    packets.forEach((packet, i) => {
      packet.t += delta * packet.speed;
      if (packet.t >= 1) {
        packet.t = 0;
        packet.committed = false;
        const next = nextSample(i);
        packet.risk = next.p;
        packet.truth = next.fraud;
      }

      // The lane is decided once, at the gate. Deciding it every frame would make packets
      // swap sides mid-flight for no reason the viewer can see.
      //
      // The exception is while the slider is being dragged, when every packet that has not
      // yet reached its zone is re-evaluated. That is not a cheat: it is the same set of
      // transactions being scored against a new cut-off, which is exactly what the slider
      // means, and watching the spray swing from one side to the other is the clearest
      // statement this page makes.
      const rescore = hud.scrubbing && packet.t < ORBIT_T;
      if ((!packet.committed && packet.t >= GATE_T) || (rescore && packet.committed)) {
        const flagged = packet.risk >= threshold;
        if (!packet.committed) {
          packet.committed = true;
          if (flagged) routed.fraud += 1;
          else routed.legit += 1;

          // Decay first, then count: the window is always the same shape however fast
          // packets happen to be arriving.
          const decay = 0.992;
          w.legit *= decay;
          w.legitFlagged *= decay;
          w.fraud *= decay;
          w.fraudMissed *= decay;
          if (packet.truth) {
            w.fraud += 1;
            if (!flagged) w.fraudMissed += 1;
          } else {
            w.legit += 1;
            if (flagged) w.legitFlagged += 1;
          }
        }
        packet.flagged = flagged;
      }

      if (packet.t < GATE_T) {
        // Inbound: down the conduit, converging on the gate. Every packet on the same
        // stream, whatever it is about to be told.
        const u = packet.t / GATE_T;
        const converge = (1 - u) * (1 - u);
        scratch.set(
          Math.cos(packet.angle) * packet.spread * converge,
          Math.sin(packet.angle) * packet.spread * converge,
          CONDUIT * (1 - u),
        );
      } else {
        const lane = packet.flagged ? lanes.fraud : lanes.legit;
        if (packet.t < ORBIT_T) {
          lane.curve.getPoint((packet.t - GATE_T) / (ORBIT_T - GATE_T), scratch);
        } else {
          // Then it simply orbits: a decision that has been made and is being held.
          const a = ((packet.t - ORBIT_T) / (1 - ORBIT_T)) * Math.PI * 2.4;
          scratch
            .copy(lane.centre)
            .addScaledVector(lane.u, Math.cos(a) * ORBIT_RADIUS)
            .addScaledVector(lane.v, Math.sin(a) * ORBIT_RADIUS);
        }
      }

      // A fraud that was not flagged: crimson, larger, and flickering, going down the
      // emerald lane. It is the single most important thing on screen at a high threshold
      // and it is drawn to be impossible to miss.
      const leaking = packet.committed && packet.truth && !packet.flagged;

      dummy.position.copy(scratch);
      dummy.scale.setScalar(
        (0.034 + packet.risk * 0.04) * (leaking ? 1.5 + Math.sin(t * 14 + i) * 0.35 : 1),
      );
      dummy.rotation.set(0, 0, 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      if (!packet.committed) colour.setRGB(0.78, 0.85, 0.94);
      else if (leaking) colour.copy(COLOR.fraud).multiplyScalar(1.5);
      else colour.copy(packet.flagged ? COLOR.fraud : COLOR.legit);
      mesh.setColorAt(i, colour);
    });

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    // Published for the vault, the post chain and the rail. Both are shares of a labelled
    // population, so neither depends on how the sample happens to be mixed.
    alarm.flood = Math.min(1, w.legit > 0.5 ? w.legitFlagged / w.legit / 0.5 : 0);
    alarm.leak = Math.min(1, w.fraud > 0.5 ? w.fraudMissed / w.fraud / 0.85 : 0);
  });

  /* ------------------------------------------------------------------ render */

  return (
    <group>
      {/* The conduit the scored stream arrives down. */}
      <mesh
        geometry={tubeGeometry}
        position={[0, 0, CONDUIT / 2]}
        rotation={[Math.PI / 2, 0, 0]}
      >
        <meshBasicMaterial
          color="#7fb0e8"
          transparent
          opacity={0.05}
          blending={AdditiveBlending}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
      {supports.map((z) => (
        <mesh key={z} geometry={supportGeometry} position={[0, 0, z]}>
          <meshBasicMaterial
            color={HEX.brand}
            transparent
            opacity={0.3}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      ))}

      <DecisionGate />
      <Label
        text="P(fraud) >= threshold"
        pill
        size={64}
        position={[0, 2.16, 0]}
        height={0.3}
        color={HEX.brandBright}
      />

      {/* Approved. */}
      <mesh geometry={laneBeams.legit}>
        <meshBasicMaterial
          color={HEX.legit}
          transparent
          opacity={0.35}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <ShieldZone centre={ZONE.legit} />
      <Label
        text="APPROVED · LOW RISK"
        pill
        size={64}
        mono={false}
        weight={700}
        position={[ZONE.legit.x, ZONE.legit.y - 2.35, ZONE.legit.z]}
        height={0.32}
        color={HEX.legit}
      />

      {/* Flagged. */}
      <mesh geometry={laneBeams.fraud}>
        <meshBasicMaterial
          color={HEX.fraud}
          transparent
          opacity={0.35}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>
      <ContainmentVault centre={ZONE.fraud} />
      <Label
        text="ALERT · HIGH RISK"
        pill
        size={64}
        mono={false}
        weight={700}
        position={[ZONE.fraud.x, ZONE.fraud.y + 2.85, ZONE.fraud.z]}
        height={0.32}
        color={HEX.fraud}
      />

      <instancedMesh
        ref={packetRef}
        args={[packetGeometry, packetMaterial, packets.length]}
        frustumCulled={false}
      />
    </group>
  );
}

/* --------------------------------------------------------------------- util */

/**
 * A thin tube along a curve.
 *
 * three's TubeGeometry is the obvious thing here and builds a full radial mesh with normals
 * and UVs; at a radius of fifteen thousandths of a unit none of that is ever seen. This is
 * the same silhouette for a twentieth of the vertices.
 */
class TubeGeometryLite extends BufferGeometry {
  constructor(curve: QuadraticBezierCurve3, segments = 44, radius = 0.016, sides = 4) {
    super();
    const positions: number[] = [];
    const indices: number[] = [];
    const point = new Vector3();
    const next = new Vector3();
    const tangent = new Vector3();
    const normal = new Vector3();
    const binormal = new Vector3();

    for (let i = 0; i <= segments; i++) {
      curve.getPoint(i / segments, point);
      curve.getPoint(Math.min(1, i / segments + 0.01), next);
      tangent.subVectors(next, point).normalize();
      normal.set(0, 1, 0).cross(tangent).normalize();
      if (normal.lengthSq() < 1e-6) normal.set(1, 0, 0);
      binormal.crossVectors(tangent, normal).normalize();

      for (let s = 0; s < sides; s++) {
        const a = (s / sides) * Math.PI * 2;
        positions.push(
          point.x + (normal.x * Math.cos(a) + binormal.x * Math.sin(a)) * radius,
          point.y + (normal.y * Math.cos(a) + binormal.y * Math.sin(a)) * radius,
          point.z + (normal.z * Math.cos(a) + binormal.z * Math.sin(a)) * radius,
        );
      }
    }

    for (let i = 0; i < segments; i++) {
      for (let s = 0; s < sides; s++) {
        const a = i * sides + s;
        const b = i * sides + ((s + 1) % sides);
        const c = (i + 1) * sides + ((s + 1) % sides);
        const d = (i + 1) * sides + s;
        indices.push(a, b, c, a, c, d);
      }
    }

    this.setAttribute("position", new Float32BufferAttribute(positions, 3));
    this.setIndex(indices);
  }
}
