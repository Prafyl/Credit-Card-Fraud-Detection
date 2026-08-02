// The landing page's WebGL stage: one world, one camera, one continuous move through it.
//
// The canvas element is deliberately not full-bleed: the page gives it the left sixty
// percent of the viewport and keeps the right forty for the reading rail. Sizing the canvas
// to its column, rather than stretching it behind the whole page and hoping z-index and
// blur keep the two apart, is what makes text-over-3D collisions structurally impossible
// instead of a thing to keep re-checking.
//
// Everything below lives in a single coordinate space. The five chapters are anchored at
// five depths along the -Z corridor, the camera flies from the first to the last without
// cutting, and the handover between chapters is the camera physically passing through one
// into the next. Fog does the rest: a chapter behind you dims because it is behind you.
//
// The rails are hand-placed. The interesting part of a camera move is where it slows down
// and what it is pointed at while it does, and neither survives being derived from a
// formula. Between those waypoints the position is smoothstepped and then critically damped,
// so the path has no corners and the picture has no jitter at any scroll speed.
import { AdaptiveDpr, Environment, Lightformer, PerspectiveCamera, Preload } from "@react-three/drei";
import { EffectComposer } from "@react-three/postprocessing";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { easing } from "maath";
import {
  BloomEffect,
  ChromaticAberrationEffect,
  DepthOfFieldEffect,
  GlitchEffect,
  GlitchMode,
  VignetteEffect,
} from "postprocessing";
import { Suspense, useEffect, useMemo, useRef } from "react";
import { Vector2, Vector3 } from "three";
import { gpuTier } from "../lib/three/capabilities";
import { alarm, flight } from "../lib/three/hudState";
import { pointer, scrollProgress, trackPointer } from "../lib/three/scrollState";
import { Chapter } from "./Chapter";
import { CHAPTERS, CHIP } from "./chapters";
import { Corridor } from "./objects/Corridor";
import { Dust } from "./objects/Dust";
import { CardScene } from "./scenes/CardScene";
import { ConsoleScene } from "./scenes/ConsoleScene";
import { DecisionCore } from "./scenes/DecisionCore";
import { VectorSpace } from "./scenes/VectorSpace";
import { VerdictScene } from "./scenes/VerdictScene";

interface Keyframe {
  at: number;
  pos: [number, number, number];
  look: [number, number, number];
}

/**
 * The flight, as waypoints on one path.
 *
 * Read the z column on its own and the shape of the story is visible in it: the camera only
 * ever moves further down the corridor, never back. Where consecutive waypoints are close in
 * z the camera is holding on something and the rail beside it has copy to read; where they
 * jump the camera is travelling, and the corridor lights up to say so.
 */
const KEYFRAMES: Keyframe[] = [
  /* 01 -- the card, held in the lower half of the column with air above it, then approached
     along its face and lined up square with the chip. */
  { at: 0.0, pos: [0.2, 0.62, 5.1], look: [0, 0.32, 0] },
  { at: 0.065, pos: [0.98, 0.5, 3.85], look: [-0.45, 0.3, 0] },
  { at: 0.125, pos: [CHIP[0] * 0.9, CHIP[1] * 0.9, 1.7], look: [CHIP[0], CHIP[1], -4] },

  /* the dive: through the contact plate and into the corridor behind it */
  { at: 0.18, pos: [CHIP[0] * 0.35, CHIP[1] * 0.35, -1.6], look: [0, 0, -16] },
  { at: 0.23, pos: [0, 0, -12.5], look: [0, 0, -28] },

  /* 02 -- the feature space, entered from the tunnel and circled once */
  { at: 0.29, pos: [0, 0.45, -22.6], look: [0, 0, -34] },
  { at: 0.345, pos: [3.3, 1.35, -25.6], look: [0, 0, -34] },
  { at: 0.395, pos: [1.1, -1.2, -30.2], look: [0, -0.2, -38] },

  /* the collapse: the cluster funnels down-corridor and the camera goes with it */
  { at: 0.45, pos: [0, 0, -48], look: [0, 0, -62] },

  /* 03 -- the core, approached head-on down the funnel and then circled */
  { at: 0.505, pos: [0, 0.85, -63.2], look: [0, 0, -76] },
  { at: 0.56, pos: [-3.1, 1.2, -66.4], look: [0, 0, -76] },
  { at: 0.615, pos: [0.9, -1.3, -71.5], look: [0, -0.15, -80] },

  /* down the conduit that carries the scored probabilities to the gate */
  { at: 0.665, pos: [0, 0.3, -92], look: [0, 0, -106] },

  /* 04 -- the gate, then a long hold with both zones in frame while the slider is worked */
  { at: 0.725, pos: [0, 1.5, -106.5], look: [0, 0.2, -118] },
  { at: 0.8, pos: [0, 0.75, -108.8], look: [0, 0, -118.4] },
  { at: 0.855, pos: [0, 0.4, -109.6], look: [0, 0, -118.6] },

  /* 05 -- past the zones and out to where the whole engine can be seen at once */
  { at: 0.905, pos: [0, 0.3, -128], look: [0, 0.1, -144] },
  { at: 0.96, pos: [0, 0.5, -143.2], look: [0, 0.3, -152] },
  { at: 1.0, pos: [0, 0.45, -143.9], look: [0, 0.3, -152.2] },
];

const scratchPos = new Vector3();
const scratchLook = new Vector3();

/** Where the camera is currently pointed, in world space. The autofocus reads this. */
const focus = new Vector3();

/** Metres either side of the focus point that stay sharp. Wide enough to hold a whole
 *  chapter, narrow enough that the corridor past it falls away. */
const FOCUS_RANGE = 16;

function sample(p: number, key: "pos" | "look", out: Vector3): Vector3 {
  const clamped = Math.min(1, Math.max(0, p));
  let a = KEYFRAMES[0];
  let b = KEYFRAMES[KEYFRAMES.length - 1];
  for (let i = 0; i < KEYFRAMES.length - 1; i++) {
    if (clamped >= KEYFRAMES[i].at && clamped <= KEYFRAMES[i + 1].at) {
      a = KEYFRAMES[i];
      b = KEYFRAMES[i + 1];
      break;
    }
  }
  const span = b.at - a.at || 1;
  const t = (clamped - a.at) / span;
  // Smoothstep between waypoints, so the camera eases through each one instead of changing
  // direction sharply at every corner of the path.
  const s = t * t * (3 - 2 * t);
  const av = a[key];
  const bv = b[key];
  return out.set(
    av[0] + (bv[0] - av[0]) * s,
    av[1] + (bv[1] - av[1]) * s,
    av[2] + (bv[2] - av[2]) * s,
  );
}

function CameraRig() {
  // damp3 mutates its target in place, so the eased look-at point has to persist across
  // frames rather than being rebuilt each time.
  const look = useRef(new Vector3());
  const drift = useRef(new Vector3());
  const previous = useRef(new Vector3(0, 0, 6.5));

  useFrame((state, delta) => {
    const p = scrollProgress.current;

    sample(p, "pos", scratchPos);
    sample(p, "look", scratchLook);

    // The stage is a tall, narrow column rather than a landscape frame, so the camera has
    // to stand further back from what it is looking at than a 16:9 shot would need to.
    const aspect = state.size.width / Math.max(1, state.size.height);
    const pull = Math.min(1.7, Math.max(0.95, 1.25 / aspect));
    scratchPos.sub(scratchLook).multiplyScalar(pull).add(scratchLook);

    // A small pointer-driven offset on top of the scripted path. It is what stops the shot
    // feeling like a video: the camera answers the mouse even while it is on rails. It is
    // wound down during fast travel, where a hand-held wobble reads as a fault.
    const settle = 1 - flight.rush * 0.75;
    easing.damp3(drift.current, [pointer.x * 0.34 * settle, pointer.y * 0.24 * settle, 0], 0.5, delta);
    scratchPos.add(drift.current);

    easing.damp3(state.camera.position, scratchPos, 0.26, delta);
    easing.damp3(look.current, scratchLook, 0.3, delta);
    state.camera.lookAt(look.current);

    focus.copy(look.current);
    flight.z = state.camera.position.z;

    // Speed is measured from where the camera actually ended up, after damping and after
    // the pointer offset -- so the corridor reacts to the picture rather than to the scroll
    // number that produced it. Twenty units a second is a brisk transit.
    const travelled = state.camera.position.distanceTo(previous.current);
    previous.current.copy(state.camera.position);
    const rush = Math.min(1, travelled / Math.max(1e-4, delta) / 20);
    // Rising quickly and falling slowly: a transition should announce itself at once and
    // then let go, rather than flickering on every small correction.
    flight.rush += (rush - flight.rush) * Math.min(1, delta * (rush > flight.rush ? 9 : 2.6));
  });

  return null;
}

/**
 * The post chain, built by hand rather than through the declarative wrappers.
 *
 * Four of these five effects have to be driven every frame from values that live outside
 * React -- camera speed, the autofocus target, the alarm state of the verdict gate -- and
 * holding the effect instances directly is both simpler and cheaper than routing sixty
 * prop changes a second through the reconciler to reach the same objects.
 */
function Effects({ tier }: { tier: "low" | "mid" | "high" }) {
  const camera = useThree((s) => s.camera);

  const fx = useMemo(() => {
    const bloom = new BloomEffect({
      intensity: 0.78,
      luminanceThreshold: 0.28,
      luminanceSmoothing: 0.86,
      mipmapBlur: true,
    });
    // Radially modulated, so the fringing is absent in the middle of frame and strongest at
    // the edges. Applied uniformly it looks like a broken display; applied radially it looks
    // like a lens.
    const chroma = new ChromaticAberrationEffect({
      offset: new Vector2(0, 0),
      radialModulation: true,
      modulationOffset: 0.4,
    });
    const vignette = new VignetteEffect({ eskil: false, offset: 0.2, darkness: 0.82 });
    const glitch = new GlitchEffect({
      delay: new Vector2(1.4, 3.2),
      duration: new Vector2(0.1, 0.26),
      strength: new Vector2(0.06, 0.22),
      ratio: 0.86,
    });
    glitch.mode = GlitchMode.DISABLED;

    // Depth of field is the most expensive thing here by some distance, so it is the one
    // effect that is earned rather than assumed.
    //
    // It is also the one that has to be driven in world units rather than through the
    // effect's own `target`. This scene runs from a card five units away to a corridor a
    // hundred and seventy units long, and the first pass at it -- constructor options plus
    // an auto-focus target -- put the whole of chapter one out of focus hard enough that a
    // dark card on a transparent canvas blurred away to nothing. The circle-of-confusion
    // material takes metres, so it is given metres, every frame, explicitly.
    const dof =
      tier === "high"
        ? new DepthOfFieldEffect(camera, { bokehScale: 1.1, resolutionScale: 0.5 })
        : null;
    if (dof) {
      dof.target = null;
      dof.cocMaterial.worldFocusRange = FOCUS_RANGE;
    }

    return { bloom, chroma, vignette, glitch, dof };
  }, [camera, tier]);

  useEffect(
    () => () => {
      fx.bloom.dispose();
      fx.chroma.dispose();
      fx.vignette.dispose();
      fx.glitch.dispose();
      fx.dof?.dispose();
    },
    [fx],
  );

  useFrame(() => {
    const rush = flight.rush;

    // Squared, so the fringing is invisible while reading and unmistakable in a dive.
    const fringe = rush * rush;
    fx.chroma.offset.set(fringe * 0.0028, fringe * 0.0017);
    fx.bloom.intensity = 0.78 + rush * 0.55;

    if (fx.dof) {
      // Focus follows what the camera is pointed at, in metres. Everything the shot is
      // actually about sits within FOCUS_RANGE of that point; the corridor beyond it goes
      // soft, which is the whole reason the effect is here.
      fx.dof.cocMaterial.worldFocusDistance = camera.position.distanceTo(focus);
      fx.dof.bokehScale = 0.9 + rush * 1.6;
    }

    // The frame only breaks up when the operating point is genuinely failing: the vault
    // flooded with false alarms, or fraud being waved through. It is a readout, not a mood.
    const stress = Math.max(alarm.flood, alarm.leak);
    const mode = stress > 0.66 ? GlitchMode.SPORADIC : GlitchMode.DISABLED;
    if (fx.glitch.mode !== mode) fx.glitch.mode = mode;
  });

  return (
    <EffectComposer multisampling={tier === "high" ? 4 : 0}>
      {fx.dof ? <primitive object={fx.dof} /> : <></>}
      <primitive object={fx.bloom} />
      <primitive object={fx.chroma} />
      <primitive object={fx.vignette} />
      <primitive object={fx.glitch} />
    </EffectComposer>
  );
}

/**
 * Emissive panels baked once into an environment map: a bright overhead softbox, cool
 * verticals either side, and one warm strip low and right.
 *
 * drei ships `<Environment preset="city" />`, which is the obvious thing to reach for here
 * and is what the brief asked for. It fetches an HDRI from a CDN at runtime, and a demo
 * that has to work on a lecture-room machine with no network would render a black,
 * unlit card. This rig is the same lighting idea, built from geometry, and it produces the
 * real cubemap the physical materials need without a single request leaving the page.
 */
function StudioEnvironment() {
  return (
    <Environment resolution={256} frames={1}>
      {/* A large, dim panel filling most of the hemisphere the camera sits in. It is what
          the flat of the card actually reflects, and without it a metalness-0.88 face has
          nothing to show and renders as a black rectangle. */}
      <Lightformer intensity={0.6} color="#8ea6c4" position={[0, 0, 9]} scale={[18, 11, 1]} />
      <Lightformer intensity={2.8} color="#ffffff" position={[0, 5, 2]} scale={[10, 3, 1]} rotation={[Math.PI / 2, 0, 0]} />
      {/* The panel that actually shows up on the card: broad, front-left, and angled so its
          reflection sweeps diagonally across the face rather than sitting as a hotspot. */}
      <Lightformer intensity={2.4} color="#dbe9ff" position={[-3.5, 2.6, 6]} scale={[7, 4.5, 1]} rotation={[0, 0.5, 0.35]} />
      <Lightformer intensity={1.9} color="#9cc6ff" position={[-6, 1, 3]} scale={[3, 9, 1]} rotation={[0, Math.PI / 2, 0]} />
      <Lightformer intensity={1.4} color="#cfe0f5" position={[6, -1, 2]} scale={[3, 7, 1]} rotation={[0, -Math.PI / 2, 0]} />
      <Lightformer intensity={2.4} color="#f5c877" position={[3.5, -3.2, 4.5]} scale={[5.5, 1.8, 1]} rotation={[0, -0.4, -0.25]} />
      <Lightformer intensity={0.7} color="#2c3d55" position={[0, 0, -8]} scale={[12, 12, 1]} />
    </Environment>
  );
}

export function CanvasRoot({ active }: { active: boolean }) {
  const tier = gpuTier();
  const maxDpr = tier === "low" ? 1 : tier === "mid" ? 1.5 : 1.75;

  useEffect(trackPointer, []);

  return (
    <Canvas
      // Rendering stops entirely once the story has scrolled past, rather than burning a
      // GPU budget on a canvas nobody can see.
      frameloop={active ? "always" : "never"}
      dpr={[1, maxDpr]}
      gl={{ antialias: tier !== "low", powerPreference: "high-performance", alpha: true }}
      camera={{ position: [0.3, 0.5, 6.5], fov: 45, near: 0.06, far: 240 }}
    >
      {/* The transition mechanism, and the reason no chapter has to be faded out by hand.
          Density is tuned to the ~40-unit spacing between anchors: the chapter you have just
          left is about half gone, the one before it is not there at all. */}
      <fogExp2 attach="fog" args={["#06070a", 0.018]} />

      <PerspectiveCamera makeDefault fov={45} near={0.06} far={240} position={[0.3, 0.5, 6.5]} />
      <CameraRig />

      <ambientLight intensity={0.32} />
      {/* Gold key from the upper right and a cool rim from behind left: the two lights that
          make brushed metal read as metal rather than as dark plastic. No fill light in
          front of the card -- a point light on the viewing axis puts a blown-out specular
          blob in the middle of the face, which reads as a rendering artefact. */}
      <directionalLight position={[5.5, 5, 4]} intensity={2.6} color="#ffd9a3" />
      <directionalLight position={[-5, 1.5, -4]} intensity={1.6} color="#8fd0ff" />

      <Suspense fallback={null}>
        <StudioEnvironment />
        <Corridor streaks={tier === "low" ? 90 : tier === "mid" ? 170 : 240} />
        <Dust count={tier === "low" ? 700 : 1800} />

        {CHAPTERS.map((chapter) => (
          <Chapter key={chapter.id} chapter={chapter}>
            {chapter.id === "card" && <CardScene />}
            {chapter.id === "vectors" && <VectorSpace />}
            {chapter.id === "core" && <DecisionCore />}
            {chapter.id === "verdict" && <VerdictScene />}
            {chapter.id === "console" && <ConsoleScene />}
          </Chapter>
        ))}

        <Preload all />
      </Suspense>

      {tier !== "low" && <Effects tier={tier} />}
      <AdaptiveDpr pixelated />
    </Canvas>
  );
}
