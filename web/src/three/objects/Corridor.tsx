// The thing that makes five chapters one shot.
//
// A single structure runs the entire length of the story: the card's own silhouette,
// repeated into the distance and opening out as it goes, with rails joining corresponding
// corners so it reads as one continuous framework rather than as a stack of loose hoops.
// The first frame sits flush with the card's edge band. When the card burns open around the
// chip, what is behind it is its own outline, carrying on -- the metal does not vanish, it
// pulls back and becomes the architecture of everywhere else you are about to go.
//
// Everything else in the story is anchored inside this framework, so a chapter never
// arrives from nowhere: it is already there, ahead of you, in the fog.
//
// The streaks are the second half of the idea. They key off actual camera speed rather than
// off scroll position, so the corridor lights up exactly when you are moving through it and
// settles the moment the camera stops to look at something. That is what sells a scroll as
// a flight instead of as a sequence of scenes.
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  BoxGeometry,
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  InstancedMesh,
  LineBasicMaterial,
  MeshBasicMaterial,
  Object3D,
} from "three";
import { flight } from "../../lib/three/hudState";
import { scrollProgress } from "../../lib/three/scrollState";
import { CARD_ASPECT, CORRIDOR_END, corridorHalfWidth, ease } from "../chapters";

/** First frame, flush behind the card, and the gap between frames after it. */
const FIRST_Z = -1.2;
const SPACING = 6;

/** Segments per rounded corner. Four is enough at the sizes these are ever seen at. */
const CORNER_STEPS = 4;

const GOLD = new Color("#d9ae55");
const BRAND = new Color("#3987e5");
const COOL = new Color("#8fd0ff");

/** One rounded-rectangle outline, in the card's proportion, as a closed loop of points. */
function frameLoop(halfWidth: number): Array<[number, number]> {
  const hw = halfWidth;
  const hh = halfWidth / CARD_ASPECT;
  const r = hw * 0.07;
  const points: Array<[number, number]> = [];

  // Corners run anticlockwise from the bottom right, each as a quarter arc between the two
  // straight runs that meet there.
  const corners: Array<[number, number, number]> = [
    [hw - r, -hh + r, -Math.PI / 2],
    [hw - r, hh - r, 0],
    [-hw + r, hh - r, Math.PI / 2],
    [-hw + r, -hh + r, Math.PI],
  ];

  for (const [cx, cy, start] of corners) {
    for (let s = 0; s <= CORNER_STEPS; s++) {
      const a = start + (s / CORNER_STEPS) * (Math.PI / 2);
      points.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
  }
  return points;
}

/**
 * The framework: every frame's outline plus the four rails that join them.
 *
 * Built once, as a single LineSegments, because a corridor is one object and drawing it as
 * one is both the truthful description and the cheap one.
 */
function buildFramework(): { geometry: BufferGeometry; frames: number } {
  const positions: number[] = [];
  const colors: number[] = [];
  const tint = new Color();

  const count = Math.floor((FIRST_Z - CORRIDOR_END) / SPACING);
  const loops: Array<Array<[number, number]>> = [];
  const depths: number[] = [];

  for (let k = 0; k < count; k++) {
    const z = FIRST_Z - k * SPACING;
    depths.push(z);
    loops.push(frameLoop(corridorHalfWidth(z)));
  }

  const colourAt = (z: number, bulkhead: boolean) => {
    // Gold at the card, cooling to the corridor's blue over the first stretch: the metal
    // becomes the architecture rather than being replaced by it.
    tint.copy(GOLD).lerp(BRAND, Math.min(1, -z / 26));
    tint.lerp(COOL, Math.min(1, Math.max(0, (-z - 70) / 90)));
    return bulkhead ? tint.clone().multiplyScalar(1.5) : tint.clone();
  };

  const push = (
    ax: number,
    ay: number,
    az: number,
    bx: number,
    by: number,
    bz: number,
    c: Color,
  ) => {
    positions.push(ax, ay, az, bx, by, bz);
    colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
  };

  loops.forEach((loop, k) => {
    const z = depths[k];
    // Every fourth frame is a bulkhead. It gives the flight a beat: without it a constant
    // stream of identical hoops has no rhythm and reads as a texture rather than as travel.
    const c = colourAt(z, k % 4 === 0);
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i];
      const b = loop[(i + 1) % loop.length];
      push(a[0], a[1], z, b[0], b[1], z, c);
    }
  });

  // Rails along the four corners of the frames, which is what turns hoops into a tunnel.
  const railIndices = [0, CORNER_STEPS + 1, (CORNER_STEPS + 1) * 2, (CORNER_STEPS + 1) * 3];
  for (let k = 0; k < loops.length - 1; k++) {
    const c = colourAt((depths[k] + depths[k + 1]) / 2, false).multiplyScalar(0.55);
    for (const i of railIndices) {
      const a = loops[k][i];
      const b = loops[k + 1][i];
      push(a[0], a[1], depths[k], b[0], b[1], depths[k + 1], c);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new Float32BufferAttribute(colors, 3));
  return { geometry, frames: loops.length };
}

/** How far ahead of and behind the camera the streaks are kept. */
const AHEAD = 7;
const BEHIND = 92;

export function Corridor({ streaks = 220 }: { streaks?: number }) {
  const framework = useMemo(buildFramework, []);
  const rush = useRef<InstancedMesh>(null);

  const dummy = useMemo(() => new Object3D(), []);
  const tint = useMemo(() => new Color(), []);

  const frameMaterial = useMemo(
    () =>
      new LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0,
        blending: AdditiveBlending,
        depthWrite: false,
      }),
    [],
  );

  // A unit-length box scaled along Z. A stretched box is what a fast-moving point looks
  // like once the shutter is open for any time at all, and it costs the same as a sphere.
  const streakGeometry = useMemo(() => new BoxGeometry(1, 1, 1), []);
  const streakMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        blending: AdditiveBlending,
        depthWrite: false,
        fog: false,
      }),
    [],
  );

  // Fixed lanes on the corridor wall. `u` is the streak's place in the window that follows
  // the camera, so the field is always around you and never has to be respawned.
  const lanes = useMemo(
    () =>
      Array.from({ length: streaks }, (_, i) => ({
        angle: (i * 2.39996323) % (Math.PI * 2),
        radius: 0.74 + ((i * 13) % 9) * 0.045,
        u: (i * 0.618033) % 1,
        speed: 0.55 + ((i * 7) % 11) * 0.075,
        thickness: 0.014 + ((i * 5) % 4) * 0.007,
      })),
    [streaks],
  );

  useEffect(
    () => () => {
      framework.geometry.dispose();
      frameMaterial.dispose();
      streakGeometry.dispose();
      streakMaterial.dispose();
    },
    [framework, frameMaterial, streakGeometry, streakMaterial],
  );

  useFrame((state, delta) => {
    const p = scrollProgress.current;

    // Absent for the opening shot -- both the framework and the streaks. A wireframe tunnel
    // behind a floating credit card is science fiction; the card has to be photographed
    // against nothing, and the architecture only exists once you are inside the chip. The
    // first pass at this gated the frames but not the streaks, and a handful of them drifted
    // through the opening shot as pale smears either side of the card.
    // Weighted almost entirely on camera speed rather than held at a constant level. The
    // corridor is a travel device: while the camera is moving it should be the loudest thing
    // on screen, and while it is holding on a chapter it has no business competing with the
    // subject. At rest the framework is barely a suggestion of walls.
    const arrived = ease(p, 0.075, 0.175);
    frameMaterial.opacity = arrived * (0.035 + flight.rush * 0.26);

    const camZ = state.camera.position.z;
    const mesh = rush.current;
    if (!mesh) return;

    const intensity = flight.rush;
    streakMaterial.opacity = arrived * (0.02 + intensity * 0.7);
    // Gold inside the chip, cooling to the corridor's own blue further down.
    tint.copy(GOLD).lerp(COOL, Math.min(1, Math.max(0, -camZ / 30)));
    streakMaterial.color.copy(tint);

    mesh.visible = streakMaterial.opacity > 0.01;
    if (!mesh.visible) return;

    const rate = 0.06 + intensity * 0.9;
    const length = 0.3 + intensity * intensity * 7.5;

    lanes.forEach((lane, i) => {
      lane.u = (lane.u + delta * rate * lane.speed) % 1;
      const z = camZ + AHEAD - lane.u * (AHEAD + BEHIND);
      const r = corridorHalfWidth(z) * lane.radius;

      dummy.position.set(Math.cos(lane.angle) * r, (Math.sin(lane.angle) * r) / CARD_ASPECT, z);
      dummy.scale.set(lane.thickness, lane.thickness, length);
      dummy.rotation.set(0, 0, 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <group>
      <lineSegments geometry={framework.geometry} material={frameMaterial} frustumCulled={false} />
      <instancedMesh
        ref={rush}
        args={[streakGeometry, streakMaterial, lanes.length]}
        frustumCulled={false}
      />
    </group>
  );
}
