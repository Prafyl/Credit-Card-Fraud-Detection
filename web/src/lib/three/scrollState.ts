// Page scroll progress, 0 at the top and 1 at the bottom.
//
// Deliberately a bare mutable object rather than React state or context: the scroll handler
// writes to it ~60x a second and the 3D camera reads it inside useFrame, so routing it
// through React would re-render the landing tree on every frame.
//
// It lives in its own module so the scroll hook can import it WITHOUT pulling in CanvasRoot
// -- otherwise the static import chain defeats the lazy() around three.js and the whole 3D
// bundle loads on first paint.
export const scrollProgress = { current: 0 };

/**
 * Pointer position in clip space, -1..1 on each axis, with y up.
 *
 * R3F already tracks this as `state.pointer`, but only from events that reach the canvas
 * element, and the canvas sits behind the page at a negative z-index with pointer events
 * switched off so it cannot swallow clicks meant for the rail. Listening on the window
 * instead is what keeps the camera answering the mouse anyway.
 */
export const pointer = { x: 0, y: 0 };

/** Starts tracking the pointer. Returns the teardown. */
export function trackPointer(): () => void {
  const onMove = (e: PointerEvent) => {
    pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
    pointer.y = -((e.clientY / window.innerHeight) * 2 - 1);
  };
  window.addEventListener("pointermove", onMove, { passive: true });
  return () => window.removeEventListener("pointermove", onMove);
}
