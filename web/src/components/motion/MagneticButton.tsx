// A button that leans toward the cursor. Small effect, but it is the sort of detail that
// separates a demo that feels built from one that feels templated.
import { motion, useMotionValue, useReducedMotion, useSpring } from "framer-motion";
import { ReactNode, useRef } from "react";

export function MagneticButton({
  children,
  onClick,
  variant = "primary",
  className = "",
  strength = 0.35,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost";
  className?: string;
  strength?: number;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const reduce = useReducedMotion();

  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 260, damping: 18, mass: 0.4 });
  const sy = useSpring(y, { stiffness: 260, damping: 18, mass: 0.4 });

  const styles =
    variant === "primary"
      ? "bg-brand text-white shadow-glow-brand hover:bg-brand/90"
      : "border border-line bg-white/5 text-ink hover:bg-white/10 backdrop-blur";

  return (
    <motion.button
      ref={ref}
      onClick={onClick}
      style={reduce ? undefined : { x: sx, y: sy }}
      onPointerMove={(e) => {
        if (reduce || !ref.current) return;
        const r = ref.current.getBoundingClientRect();
        x.set((e.clientX - (r.left + r.width / 2)) * strength);
        y.set((e.clientY - (r.top + r.height / 2)) * strength);
      }}
      onPointerLeave={() => {
        x.set(0);
        y.set(0);
      }}
      whileTap={{ scale: 0.96 }}
      className={`inline-flex items-center justify-center gap-2 rounded-xl px-7 py-3.5 text-base font-medium transition-colors ${styles} ${className}`}
    >
      {children}
    </motion.button>
  );
}
