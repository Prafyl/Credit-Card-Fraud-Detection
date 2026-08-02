// Chapter 1: the card.
//
// A physical object, built the way the object is actually built: a brushed titanium slab
// with a polished edge band, foil stamped into its face, a gold contact plate, and the
// contactless mark beside it. Nothing here is a glowing rectangle standing in for a card.
//
// The exit is a shader dissolve rather than a fade. Fading would leave the camera drifting
// through an empty rectangle where a card used to be; discarding fragments inside a circle
// that grows from the chip means there is a real hole with a hot rim, and the metal stays
// solid right up to its cut edge while the camera flies through it.
import { RoundedBox } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import {
  AdditiveBlending,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  RepeatWrapping,
  Sprite,
  SpriteMaterial,
} from "three";
import { brushedFinish, cardPrintTexture, glowSprite } from "../../lib/three/cardTextures";
import { pointer, scrollProgress } from "../../lib/three/scrollState";
import { CHIP, ease, range } from "../chapters";

/* 85.60 x 53.98 mm, ISO/IEC 7810 ID-1, at 1 world unit = 25 mm. */
const W = 3.424;
const H = 2.159;
const D = 0.052;

/** Where the chip's own geometry sits on the face. */
const CHIP_W = 0.5;
const CHIP_H = 0.42;
const FACE_Z = D / 2;

/** Scroll span over which the card burns open around the chip. It has to be finished by the
 *  time the camera reaches the card's plane, which the flight does at about 0.15. */
const DISSOLVE: [number, number] = [0.108, 0.172];

/** How far the edge band trails the face. See applyDissolve. */
const BAND_LAG = 0.78;

/** Radius the hole must reach to have erased every corner of the card, including the band
 *  once its lag is taken into account. */
const MAX_HOLE = Math.hypot(W / 2 + Math.abs(CHIP[0]), H / 2 + Math.abs(CHIP[1])) / BAND_LAG;

const PRINT = {
  holder: "PRAFUL BHATT",
  pan: "4291 •••• •••• 8842",
  expiry: "09/28",
  brand: "SENTINEL",
  tagline: "REAL-TIME FRAUD ENGINE",
};

/* ------------------------------------------------------------------ shader */

/**
 * Patches a material so fragments inside a circle of radius `uRadius`, centred on the chip,
 * are discarded, with a hot rim just outside the cut.
 *
 * The uniform object is passed in rather than created here so the card body, the edge band
 * and the printed face all share one radius and open as a single piece of metal.
 *
 * `lag` slows one layer's share of that radius. The polished edge band is given 0.78, so the
 * card's outline is the last part of it to go: the face burns away, the silhouette holds for
 * a beat longer, and what is behind it is the same silhouette repeated down the corridor.
 * The metal does not disappear so much as pull back and become the architecture.
 */
function applyDissolve(
  material: MeshPhysicalMaterial | MeshStandardMaterial,
  uRadius: { value: number },
  lag = 1,
) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uHole = { value: CHIP };
    shader.uniforms.uRadius = uRadius;
    shader.uniforms.uLag = { value: lag };

    // The card is a flat slab, so its two in-plane object coordinates are all the cut needs.
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec2 vCardXY;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvCardXY = position.xy;");

    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
         varying vec2 vCardXY;
         uniform vec2 uHole;
         uniform float uRadius;
         uniform float uLag;`,
      )
      // Discarding before anything else is shaded skips all the lighting work for the hole.
      .replace(
        "#include <clipping_planes_fragment>",
        `#include <clipping_planes_fragment>
         float cut = uRadius * uLag;
         float holeDist = length(vCardXY - uHole);
         if (holeDist < cut) discard;`,
      )
      .replace(
        "#include <emissivemap_fragment>",
        `#include <emissivemap_fragment>
         float rim = 1.0 - smoothstep(0.0, 0.26, holeDist - cut);
         totalEmissiveRadiance += mix(vec3(0.95, 0.70, 0.28), vec3(0.30, 0.62, 1.00), 0.35)
                                  * rim * step(0.004, cut) * 5.0;`,
      );
  };
  material.needsUpdate = true;
}

/* -------------------------------------------------------------- components */

/**
 * The EMV contact plate.
 *
 * Six gold contacts separated by etched grooves, with a finer pair of micro-etch lines
 * inside each one. The grooves are dark geometry laid on the plate rather than cut out of
 * it -- a real groove is only ever visible because it is in shadow, and a near-black strip
 * of the same metal is that, for a hundredth of the vertices.
 */
function ContactPlate() {
  const gold = useMemo(
    () =>
      new MeshPhysicalMaterial({
        color: "#c8a24e",
        metalness: 1,
        roughness: 0.19,
        clearcoat: 0.6,
        clearcoatRoughness: 0.25,
        envMapIntensity: 1.7,
      }),
    [],
  );
  const groove = useMemo(
    () => new MeshStandardMaterial({ color: "#2a2113", metalness: 0.8, roughness: 0.55 }),
    [],
  );
  const etch = useMemo(
    () => new MeshStandardMaterial({ color: "#8d6f2f", metalness: 1, roughness: 0.4 }),
    [],
  );

  useEffect(
    () => () => {
      gold.dispose();
      groove.dispose();
      etch.dispose();
    },
    [gold, groove, etch],
  );

  const top = 0.013;

  return (
    <group position={[CHIP[0], CHIP[1], FACE_Z + 0.004]}>
      <RoundedBox args={[CHIP_W, CHIP_H, 0.024]} radius={0.045} smoothness={4} material={gold} />

      {/* Contact separators: one down the middle, two across, dividing the plate into six. */}
      <mesh position={[0, 0, top]} material={groove}>
        <boxGeometry args={[0.019, CHIP_H - 0.05, 0.005]} />
      </mesh>
      {[0.068, -0.068].map((y) => (
        <mesh key={y} position={[0, y, top]} material={groove}>
          <boxGeometry args={[CHIP_W - 0.055, 0.017, 0.005]} />
        </mesh>
      ))}

      {/* Two micro-etch lines inside each of the six contacts, on the contact's own axis. */}
      {[-0.115, 0.115].map((x) =>
        [0.161, 0.117, 0.022, -0.022, -0.117, -0.161].map((y) => (
          <mesh key={`${x}:${y}`} position={[x, y, top]} material={etch}>
            <boxGeometry args={[CHIP_W * 0.3, 0.0045, 0.003]} />
          </mesh>
        )),
      )}
    </group>
  );
}

/** The contactless mark: three arcs and a terminal dot, in brushed steel. */
function ContactlessMark() {
  const material = useMemo(
    () => new MeshPhysicalMaterial({ color: "#cfd6de", metalness: 1, roughness: 0.28, envMapIntensity: 1.4 }),
    [],
  );
  useEffect(() => () => material.dispose(), [material]);

  const arc = Math.PI * 0.52;

  return (
    <group position={[-0.4, CHIP[1], FACE_Z + 0.004]}>
      {[0.075, 0.128, 0.181].map((r, i) => (
        <mesh key={r} rotation={[0, 0, -arc / 2]} material={material}>
          <torusGeometry args={[r, 0.0105 - i * 0.0012, 8, 40, arc]} />
        </mesh>
      ))}
      <mesh position={[-0.03, 0, 0]} material={material}>
        <sphereGeometry args={[0.017, 12, 12]} />
      </mesh>
    </group>
  );
}

/** Light behind the chip, and the ring that marks the edge of the opening. */
function ChipPortal({ open }: { open: React.MutableRefObject<number> }) {
  const ring = useRef<Mesh>(null);
  const core = useRef<Sprite>(null);

  const glow = useMemo(
    () =>
      new SpriteMaterial({
        map: glowSprite(),
        transparent: true,
        opacity: 0,
        blending: AdditiveBlending,
        depthWrite: false,
        fog: false,
      }),
    [],
  );
  useEffect(() => () => glow.dispose(), [glow]);

  useFrame((_, delta) => {
    const t = open.current;
    if (core.current) {
      // Grows and brightens as the opening does. It is a sprite with a quartic falloff baked
      // into it rather than a disc of flat colour: flat colour at this size is a khaki circle
      // pasted over the shot at any opacity, and bloom wrapping its hard rim in a halo only
      // makes the edge more obvious, not less.
      glow.userData.baseOpacity = 0.2 + t * 0.8;
      core.current.scale.setScalar(0.9 + t * 2.6);
    }
    if (ring.current) {
      ring.current.scale.setScalar(0.4 + t * 4.5);
      (ring.current.material as MeshBasicMaterial).userData.baseOpacity =
        Math.sin(Math.min(1, t) * Math.PI) * 0.9;
      ring.current.rotation.z += delta * 0.5;
    }
  });

  return (
    <group position={[CHIP[0], CHIP[1], FACE_Z - 0.03]}>
      <sprite ref={core} material={glow} />
      <mesh ref={ring} position={[0, 0, 0.09]}>
        <ringGeometry args={[0.3, 0.335, 64]} />
        <meshBasicMaterial
          color="#8fc4ff"
          transparent
          opacity={0}
          blending={AdditiveBlending}
          depthWrite={false}
          side={DoubleSide}
        />
      </mesh>
    </group>
  );
}

/* ------------------------------------------------------------------- scene */

export function CardScene() {
  const group = useRef<Group>(null);
  const chip = useRef<Group>(null);
  const open = useRef(0);

  // One uniform object, shared by every material that has to be cut by the same hole.
  const uRadius = useRef({ value: 0 });

  const finish = useMemo(brushedFinish, []);
  const print = useMemo(() => cardPrintTexture(PRINT), []);

  // Built in one memo, with the dissolve patched in before the material is ever handed to
  // the renderer -- doing it from an effect would compile the shader twice.
  const { body, edge, face } = useMemo(() => {
    const cut = uRadius.current;

    const slab = new MeshPhysicalMaterial({
      color: "#0b0d10",
      metalness: 0.88,
      roughness: 0.3,
      roughnessMap: finish.roughness,
      normalMap: finish.normal,
      clearcoat: 1,
      clearcoatRoughness: 0.12,
      // A metal's diffuse colour is black: everything you see on this face is reflected
      // environment, so this number is the one that decides whether it reads as titanium.
      envMapIntensity: 2.0,
    });
    slab.normalScale.set(0.34, 0.34);
    // The brush runs the length of the card, so the map is stretched across it.
    for (const map of [finish.roughness, finish.normal]) {
      map.wrapS = map.wrapT = RepeatWrapping;
      map.repeat.set(1.6, 1);
    }

    // A polished band a hair proud of the matte face on every side. It is the only part of
    // the card with a mirror finish, which is what gives the silhouette a hard edge.
    const band = new MeshPhysicalMaterial({
      color: "#7f868f",
      metalness: 1,
      roughness: 0.08,
      envMapIntensity: 2.1,
    });

    const print2 = new MeshStandardMaterial({
      map: print,
      transparent: true,
      metalness: 0.92,
      roughness: 0.26,
      envMapIntensity: 1.5,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
    });

    applyDissolve(slab, cut);
    applyDissolve(band, cut, BAND_LAG);
    applyDissolve(print2, cut);

    return { body: slab, edge: band, face: print2 };
  }, [finish, print]);

  useEffect(
    () => () => {
      body.dispose();
      edge.dispose();
      face.dispose();
    },
    [body, edge, face],
  );

  useFrame((state, delta) => {
    const p = scrollProgress.current;
    const cut = range(p, DISSOLVE[0], DISSOLVE[1]);
    open.current = cut;
    uRadius.current.value = cut * MAX_HOLE;

    if (group.current) {
      // Idle float and a small parallax tilt toward the pointer, both winding down as the
      // card opens so the camera is not chasing a moving target while it flies in.
      const calm = 1 - cut;
      const t = state.clock.elapsedTime;
      const k = Math.min(1, delta * 2.6);
      group.current.position.y = Math.sin(t * 0.55) * 0.07 * calm;
      group.current.rotation.x +=
        ((pointer.y * 0.2 + Math.sin(t * 0.37) * 0.045) * calm - group.current.rotation.x) * k;
      group.current.rotation.y +=
        ((pointer.x * 0.3 + Math.cos(t * 0.29) * 0.06) * calm - group.current.rotation.y) * k;
    }

    // The chip comes to meet the camera and opens out around it, so the dive ends with the
    // contact plate passing either side of frame rather than with the near plane slicing
    // through it. The scale has to outrun the camera: at the moment they cross, the plate
    // must already be wider than the path the camera is taking through it.
    //
    // And then it has to stop being there. A gold plate is solid metal, so scaling it up
    // without fading it just fills the frame with a gold wall for the last third of the
    // dive. Fading it out over the same stretch turns that wall into the contacts rushing
    // past and dissolving into the light coming through the opening behind them.
    if (chip.current) {
      chip.current.scale.setScalar(1 + cut * 5.2);
      chip.current.position.z = cut * 2.4;

      const vanish = 1 - ease(cut, 0.42, 0.88);
      chip.current.traverse((object) => {
        const material = (object as Mesh).material;
        if (!material) return;
        for (const m of Array.isArray(material) ? material : [material]) {
          m.transparent = true;
          m.depthWrite = vanish > 0.995;
          m.userData.baseOpacity = vanish;
        }
      });
    }
  });

  return (
    <group ref={group}>
      {/* Polished edge band: larger in plane, thinner through, so only its rim shows. */}
      <RoundedBox
        args={[W + 0.03, H + 0.03, D * 0.72]}
        radius={0.115}
        smoothness={6}
        material={edge}
      />

      <RoundedBox args={[W, H, D]} radius={0.1} smoothness={8} material={body} />

      <mesh position={[0, 0, FACE_Z + 0.0035]} material={face}>
        <planeGeometry args={[W, H]} />
      </mesh>

      <ChipPortal open={open} />

      <group ref={chip}>
        <ContactPlate />
        <ContactlessMark />
      </group>
    </group>
  );
}
