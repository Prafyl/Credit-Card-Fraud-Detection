// Inertial smooth scrolling, plus the 0..1 progress value the camera flight is bound to.
//
// Progress is measured across the *story* element rather than the whole document. The
// landing page continues past the 3D sequence into ordinary content, and measuring against
// document height would stretch the four-act camera move across those sections too, so the
// verdict would only arrive at the very bottom of the page.
//
// The value is written into a plain module object rather than React state on purpose: a
// setState per scroll event would re-render the entire landing tree sixty times a second.
import { RefObject, useEffect, useRef } from "react";
import Lenis from "lenis";
import { prefersReducedMotion } from "../lib/three/capabilities";
import { scrollProgress } from "../lib/three/scrollState";
import { chapterAt } from "../three/chapters";

export interface StoryState {
  /** Index into CHAPTERS, or -1 once the story has scrolled out of view. */
  chapter: number;
  /** Whether any part of the story is on screen; drives the canvas frameloop. */
  active: boolean;
}

export function useStoryScroll(
  ref: RefObject<HTMLElement>,
  onChange: (state: StoryState) => void,
  enabled = true,
) {
  // Held in a ref so a new callback identity does not tear down Lenis and restart scrolling.
  const callback = useRef(onChange);
  callback.current = onChange;

  useEffect(() => {
    if (!enabled) return;

    const reduce = prefersReducedMotion();

    // Native scrolling for reduced-motion users: smoothing is exactly the kind of motion
    // that setting is asking us to stop doing.
    const lenis = reduce
      ? null
      : new Lenis({ duration: 1.05, smoothWheel: true, touchMultiplier: 1.6 });

    // Layout is read on resize only. Calling getBoundingClientRect inside the animation
    // loop would force a synchronous layout on every frame.
    let top = 0;
    let height = 0;
    const measure = () => {
      const el = ref.current;
      if (!el) return;
      top = el.getBoundingClientRect().top + window.scrollY;
      height = el.offsetHeight;
    };

    let last: StoryState = { chapter: -2, active: false };
    const update = () => {
      const y = window.scrollY;
      const travel = Math.max(1, height - window.innerHeight);
      scrollProgress.current = Math.min(1, Math.max(0, (y - top) / travel));

      const active = y + window.innerHeight > top && y < top + height;
      const chapter = active ? chapterAt(scrollProgress.current) : -1;
      if (chapter !== last.chapter || active !== last.active) {
        last = { chapter, active };
        callback.current(last);
      }
    };

    let frame = 0;
    const raf = (time: number) => {
      lenis?.raf(time);
      update();
      frame = requestAnimationFrame(raf);
    };

    measure();
    update();
    frame = requestAnimationFrame(raf);

    const onResize = () => {
      measure();
      update();
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      lenis?.destroy();
      scrollProgress.current = 0;
    };
  }, [enabled, ref]);
}
