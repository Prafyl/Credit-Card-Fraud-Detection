// Chapter 2: the transaction as the model sees it.
//
// Thirty-one nodes, one per column of the dataset, on a shell around the thing being
// predicted. The layout carries information rather than decorating: a node's distance from
// Class at the centre is its permutation importance, so V14 and V4 -- the two columns
// holding almost all of the signal -- sit closest in and burn gold, and the columns that
// contribute nothing drift out to the edge of the shell.
//
// The chapter is entered and left through the corridor rather than cut to. Every node begins
// life at a single point up-corridor -- the far side of the chip the camera has just flown
// through -- and travels out to its place on the shell as you approach, so the constellation
// is seen to come out of the card's own circuitry. At the other end they are drawn back down
// the axis into the mouth of the decision core, which is where the camera goes next. Nothing
// in this chapter appears or disappears; it arrives and it leaves.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  SphereGeometry,
  Vector3,
} from "three";
import { FEATURES, Feature, GROUPS, importance } from "../../lib/data/features";
import { hud } from "../../lib/three/hudState";
import { COLOR, HEX } from "../../lib/three/palette";
import { scrollProgress } from "../../lib/three/scrollState";
import { CHAPTERS, ease, localProgress } from "../chapters";
import { Label } from "../objects/Label";

/** Columns whose permutation importance earns them a gold badge and a halo. */
const DRIVERS = ["V14", "V4"];

/** Up-corridor point the whole cluster unfolds from: the back of the chip. */
const SOURCE = new Vector3(0, 0, 15.5);
/** Down-corridor point it is drawn into: the intake of the decision core. */
const SINK = new Vector3(0, 0, -24);

/** Scroll spans for the unfold and the collapse. Both sit inside the chapter's own window. */
const UNFOLD: [number, number] = [0.19, 0.3];
const COLLAPSE: [number, number] = [0.4, 0.475];

interface Node {
  feature: Feature;
  /** Where the node lives once it has arrived. */
  home: Vector3;
  scale: number;
  color: Color;
  /** Which HUD toggle groups this node belongs to. */
  groups: string[];
  /** Phase offset, so the cluster unfolds as a wave rather than all at once. */
  phase: number;
}

function buildNodes(): Node[] {
  const model = FEATURES.filter((f) => f.kind === "pca" || f.kind === "amount");
  const nodes: Node[] = [];

  // Fibonacci sphere: the only placement that spaces N points evenly on a shell without
  // them lining up into visible rings or spokes.
  const n = model.length;
  model.forEach((feature, i) => {
    const imp = importance(feature);
    const y = 1 - (i / (n - 1)) * 2;
    const ring = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = i * 2.39996323;
    const radius = 2.0 + (1 - imp) * 2.35;

    nodes.push({
      feature,
      home: new Vector3(Math.cos(theta) * ring, y * 0.86, Math.sin(theta) * ring).multiplyScalar(
        radius,
      ),
      scale: 0.085 + imp * 0.24,
      color: new Color().copy(COLOR.brand).lerp(COLOR.gold, imp),
      groups: GROUPS.filter((g) => g.test(feature)).map((g) => g.id),
      // The columns that matter arrive first. It is a half-second of ordering and it makes
      // the unfold read as a ranking rather than as a firework.
      phase: (1 - imp) * 0.42,
    });
  });

  // Time is off to one side and cold: it is in the CSV but never reaches the model.
  nodes.push({
    feature: FEATURES.find((f) => f.name === "Time")!,
    home: new Vector3(3.9, 2.35, 1.1),
    scale: 0.1,
    color: new Color("#3f454d"),
    groups: ["raw"],
    phase: 0.5,
  });

  // Class sits at the centre. Everything is arranged around it because it is the thing
  // being predicted, and nothing on the shell is an input to anything but it.
  nodes.push({
    feature: FEATURES.find((f) => f.name === "Class")!,
    home: new Vector3(0, 0, 0),
    scale: 0.4,
    color: new Color(HEX.legit),
    groups: [],
    phase: 0,
  });

  return nodes;
}

/**
 * Twelve strongest columns wired to Class, plus a mesh over the surface of the shell.
 *
 * The surface mesh joins each node to its two nearest *neighbours*, not to the next two in
 * the array. Index order on a Fibonacci sphere jumps to the far side every step, so joining
 * by index draws chords straight through the middle of the cluster and the whole thing
 * turns into a ball of string. Joining by distance draws a skin.
 */
function buildEdges(nodes: Node[]): Array<[number, number]> {
  const classIndex = nodes.length - 1;
  const shell = nodes.length - 2;
  const seen = new Set<string>();
  const edges: Array<[number, number]> = [];

  const add = (a: number, b: number) => {
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push([a, b]);
  };

  for (let i = 0; i < shell; i++) {
    const nearest = [];
    for (let j = 0; j < shell; j++) {
      if (i !== j) nearest.push({ j, d: nodes[i].home.distanceTo(nodes[j].home) });
    }
    nearest.sort((a, b) => a.d - b.d);
    for (const { j } of nearest.slice(0, 2)) add(i, j);
  }

  const strongest = nodes
    .slice(0, shell)
    .map((node, i) => ({ i, imp: importance(node.feature) }))
    .sort((a, b) => b.imp - a.imp)
    .slice(0, 12);
  for (const { i } of strongest) add(i, classIndex);

  return edges;
}

export function VectorSpace() {
  const nodes = useMemo(buildNodes, []);
  const edges = useMemo(() => buildEdges(nodes), [nodes]);

  const shell = useRef<InstancedMesh>(null);
  const pulses = useRef<InstancedMesh>(null);
  const lattice = useRef<LineSegments>(null);
  const halos = useRef<Mesh[]>([]);
  const badges = useRef<Group[]>([]);
  const spin = useRef<Mesh>(null);

  /**
   * The columns worth naming, as indices into `nodes`.
   *
   * Badges only for the handful the model actually leans on: thirty-one floating strings
   * would be noise. They ride on the live positions and are scaled away during the unfold
   * and the collapse, because a label parked where a node used to be is exactly the kind of
   * leftover that makes a scene look like it is decaying rather than moving.
   */
  const badged = useMemo(() => {
    const shellCount = nodes.length - 2;
    const top = nodes
      .slice(0, shellCount)
      .map((node, i) => ({ node, i }))
      .sort((a, b) => b.node.feature.perm - a.node.feature.perm)
      .slice(0, 8);
    return [...top, { node: nodes[shellCount], i: shellCount }, { node: nodes[shellCount + 1], i: shellCount + 1 }];
  }, [nodes]);

  const driverIndices = useMemo(
    () => DRIVERS.map((name) => nodes.findIndex((n) => n.feature.name === name)),
    [nodes],
  );

  const dummy = useMemo(() => new Object3D(), []);
  const scratch = useMemo(() => new Color(), []);
  const groupKey = useRef("");

  /**
   * Where every node is this frame. Recomputed once and read by the instances, the lattice
   * and the packets, so the three can never disagree about where a node is.
   */
  const live = useMemo(() => nodes.map(() => new Vector3()), [nodes]);

  /* ---------------------------------------------------------------- geometry */

  const nodeGeometry = useMemo(() => new IcosahedronGeometry(1, 1), []);
  const nodeMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        transparent: true,
        opacity: 0.95,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const lineGeometry = useMemo(() => {
    const positions = new Float32Array(edges.length * 6);
    const colors = new Float32Array(edges.length * 6);
    edges.forEach(([a, b], i) => {
      colors.set([nodes[a].color.r, nodes[a].color.g, nodes[a].color.b], i * 6);
      colors.set([nodes[b].color.r, nodes[b].color.g, nodes[b].color.b], i * 6 + 3);
    });
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(positions, 3));
    g.setAttribute("color", new Float32BufferAttribute(colors, 3));
    return g;
  }, [edges, nodes]);

  const lineMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.1,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  const pulseGeometry = useMemo(() => new SphereGeometry(1, 8, 8), []);
  const pulseMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: HEX.brandBright,
        transparent: true,
        opacity: 0.9,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // One packet per edge, each starting at a different point along it, so the space reads as
  // continuously carrying data rather than blinking in unison.
  const packets = useMemo(
    () => edges.map((_, i) => ({ edge: i, t: (i * 0.618) % 1, speed: 0.13 + ((i * 7) % 5) * 0.045 })),
    [edges],
  );

  useEffect(
    () => () => {
      nodeGeometry.dispose();
      nodeMaterial.dispose();
      lineGeometry.dispose();
      lineMaterial.dispose();
      pulseGeometry.dispose();
      pulseMaterial.dispose();
    },
    [nodeGeometry, nodeMaterial, lineGeometry, lineMaterial, pulseGeometry, pulseMaterial],
  );

  /* ------------------------------------------------------------------ frames */

  const chapter = CHAPTERS[1];

  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const p = scrollProgress.current;
    const local = localProgress(chapter, p);

    const collapse = ease(p, COLLAPSE[0], COLLAPSE[1]);

    // Node positions first: everything else this frame is derived from them.
    nodes.forEach((node, i) => {
      // Each node's own window inside the unfold, so the cluster opens as a wave.
      const arrive = ease(p, UNFOLD[0] + node.phase * 0.05, UNFOLD[1] + node.phase * 0.05);
      const v = live[i];
      v.copy(SOURCE).lerp(node.home, arrive);

      if (collapse > 0.001) {
        // Drawn down the axis with a twist, so the cluster reads as being pulled into the
        // core rather than as shrinking on the spot.
        const spinAngle = collapse * 2.6;
        const c = Math.cos(spinAngle);
        const s = Math.sin(spinAngle);
        const x = v.x * c - v.y * s;
        const y = v.x * s + v.y * c;
        v.set(x, y, v.z).lerp(SINK, collapse * collapse);
      }
    });

    const mesh = shell.current;
    if (mesh) {
      // Re-tint only when the toggles actually change. The matrices still update every
      // frame because the nodes breathe, but the colour upload does not need to.
      const key = GROUPS.map((g) => (hud.groups[g.id] ? "1" : "0")).join("");
      const retint = key !== groupKey.current;
      groupKey.current = key;

      nodes.forEach((node, i) => {
        const on = node.groups.length === 0 || node.groups.some((g) => hud.groups[g]);
        const imp = importance(node.feature);
        const breathe = 1 + Math.sin(t * (1.1 + imp * 2.2) + i) * 0.11 * (0.35 + imp);

        dummy.position.copy(live[i]);
        dummy.scale.setScalar(node.scale * breathe * (on ? 1 : 0.45) * (1 - collapse * 0.55));
        dummy.rotation.set(t * 0.14 + i, t * 0.19 + i, 0);
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);

        if (retint || i === nodes.length - 1) {
          // Class alternates between the two outcomes it stands for.
          const color =
            i === nodes.length - 1
              ? scratch.copy(COLOR.legit).lerp(COLOR.fraud, (Math.sin(t * 0.8) + 1) / 2)
              : scratch.copy(node.color);
          mesh.setColorAt(i, color.multiplyScalar(on ? 1 : 0.3));
        }
      });

      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }

    // The lattice is rewritten from the live positions, so the wiring travels with the nodes
    // instead of the cluster tearing away from its own edges while it moves.
    const attribute = lineGeometry.getAttribute("position") as BufferAttribute;
    const array = attribute.array as Float32Array;
    edges.forEach(([a, b], i) => {
      const pa = live[a];
      const pb = live[b];
      array[i * 6 + 0] = pa.x;
      array[i * 6 + 1] = pa.y;
      array[i * 6 + 2] = pa.z;
      array[i * 6 + 3] = pb.x;
      array[i * 6 + 4] = pb.y;
      array[i * 6 + 5] = pb.z;
    });
    attribute.needsUpdate = true;

    const pulseMesh = pulses.current;
    if (pulseMesh) {
      packets.forEach((packet, i) => {
        packet.t = (packet.t + delta * packet.speed) % 1;
        const [a, b] = edges[packet.edge];
        dummy.position.lerpVectors(live[a], live[b], packet.t);
        // Fades in and out at the ends, so packets appear to enter and leave a node.
        dummy.scale.setScalar(0.03 * Math.sin(packet.t * Math.PI));
        dummy.rotation.set(0, 0, 0);
        dummy.updateMatrix();
        pulseMesh.setMatrixAt(i, dummy.matrix);
      });
      pulseMesh.instanceMatrix.needsUpdate = true;
    }

    // The whole shell turns slowly across the chapter, so the cluster is read from more
    // than one angle without the camera having to travel.
    if (lattice.current) lattice.current.rotation.y = t * 0.03 + local * 0.5;
    if (spin.current) {
      spin.current.rotation.y = -t * 0.02;
      spin.current.rotation.x = t * 0.014;
      spin.current.scale.setScalar(1 - collapse * 0.4);
    }

    // The two drivers pulse harder than the rest of the shell breathes. They are the only
    // columns whose removal costs the model anything, and the picture should say so before
    // the panel beside it does.
    halos.current.forEach((halo, i) => {
      if (!halo) return;
      halo.position.copy(live[driverIndices[i]]);
      const beat = Math.max(0, Math.sin(t * 2.1 + i * 1.7));
      halo.scale.setScalar(1 + beat * 0.3);
      (halo.material as MeshBasicMaterial).userData.baseOpacity = 0.05 + beat * 0.09;
    });

    // Badges ride their node and are only up while the cluster is still: they arrive once
    // the unfold has settled and are gone before the collapse starts moving things.
    const named = ease(p, 0.262, 0.298) * (1 - ease(p, 0.378, 0.408));
    badges.current.forEach((group, i) => {
      if (!group) return;
      group.position.copy(live[badged[i].i]);
      group.scale.setScalar(named);
    });
  });

  /* ------------------------------------------------------------------ render */

  return (
    <group>
      <instancedMesh
        ref={shell}
        args={[nodeGeometry, nodeMaterial, nodes.length]}
        frustumCulled={false}
      />
      <lineSegments ref={lattice} args={[lineGeometry, lineMaterial]} frustumCulled={false} />
      <instancedMesh
        ref={pulses}
        args={[pulseGeometry, pulseMaterial, packets.length]}
        frustumCulled={false}
      />

      {/* The boundary of the space, so the cluster reads as contained rather than adrift. */}
      <mesh ref={spin}>
        <icosahedronGeometry args={[7.2, 1]} />
        <meshBasicMaterial
          color={HEX.brand}
          wireframe
          transparent
          // Bloom multiplies this by a lot. Anything above about 0.02 stops being a hint of
          // an enclosure and becomes the brightest thing on screen.
          opacity={0.018}
          blending={AdditiveBlending}
          depthWrite={false}
        />
      </mesh>

      {/* The two columns permutation importance actually singles out. */}
      {driverIndices.map((index, i) => (
        <mesh
          key={nodes[index].feature.name}
          ref={(m) => {
            if (m) halos.current[i] = m;
          }}
        >
          <sphereGeometry args={[nodes[index].scale * 1.85, 16, 16]} />
          <meshBasicMaterial
            color={HEX.gold}
            transparent
            opacity={0.07}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      ))}

      {badged.map(({ node }, i) => {
        const driver = DRIVERS.includes(node.feature.name);
        const target = node.feature.kind === "target";
        return (
          <group
            key={node.feature.name}
            ref={(g) => {
              if (g) badges.current[i] = g;
            }}
            scale={0}
          >
            <Label
              text={node.feature.name}
              pill
              position={[0, node.scale + (target ? 0.92 : 0.3), 0]}
              height={target ? 0.34 : driver ? 0.3 : 0.24}
              color={target ? HEX.legit : driver ? HEX.gold : "#9fb4cc"}
              size={72}
            />
          </group>
        );
      })}
    </group>
  );
}
