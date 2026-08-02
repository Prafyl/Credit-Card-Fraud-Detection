// Where each chapter stands in the corridor, and the only fading anything gets.
//
// Scenes are authored around their own origin and know nothing about the rest of the story.
// This wrapper is what puts them at their anchor on the -Z axis, so the whole page is one
// coordinate space and the camera can simply fly through it.
//
// It deliberately does *not* scale scenes in and out any more. Growing a chapter from small
// while fading it up is the tell of a slideshow: it says the object was assembled in front
// of you rather than approached. Distance and fog now do that work, and they do it for the
// same reason a real lens does -- the thing is far away.
//
// A scene that wants to animate a material's opacity for itself writes to
// `material.userData.baseOpacity` instead of `material.opacity`. This wrapper's useFrame
// runs after its children's (R3F subscribes in layout-effect order, which is child first),
// so anything written straight to `opacity` would be overwritten here every frame.
import { useFrame } from "@react-three/fiber";
import { ReactNode, useRef } from "react";
import { Group, Material, Mesh } from "three";
import { scrollProgress } from "../lib/three/scrollState";
import { Chapter as ChapterDef, presenceOf } from "./chapters";

interface Tracked {
  material: Material;
  /** Opacity the material was authored with, before any fading. */
  base: number;
  /** Whether it was already in the transparent pass, so opaque geometry can be put back. */
  wasTransparent: boolean;
}

export function Chapter({ chapter, children }: { chapter: ChapterDef; children: ReactNode }) {
  const group = useRef<Group>(null);
  const tracked = useRef<Tracked[]>([]);
  const childCount = useRef(-1);

  useFrame(() => {
    const g = group.current;
    if (!g) return;

    const presence = presenceOf(chapter, scrollProgress.current);

    // Chapters outside their window cost one boolean per frame and nothing else.
    const visible = presence > 0.002;
    if (g.visible !== visible) g.visible = visible;
    if (!visible) return;

    // The scene graph is static once mounted, so the material list is gathered once. The
    // child-count guard catches the one case that is not static: a scene still mounting on
    // the frame this first runs.
    if (childCount.current !== g.children.length) {
      childCount.current = g.children.length;
      const list: Tracked[] = [];
      g.traverse((object) => {
        const material = (object as Mesh).material;
        if (!material) return;
        for (const m of Array.isArray(material) ? material : [material]) {
          list.push({ material: m, base: m.opacity, wasTransparent: m.transparent });
        }
      });
      tracked.current = list;
    }

    const full = presence > 0.999;
    for (const t of tracked.current) {
      const base = (t.material.userData.baseOpacity as number | undefined) ?? t.base;
      t.material.opacity = base * presence;
      t.material.transparent = full ? t.wasTransparent : true;
    }
  });

  return (
    <group ref={group} position={[0, 0, chapter.z]}>
      {children}
    </group>
  );
}
