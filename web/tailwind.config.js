/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Validated data-viz palette (see src/plotting.py). Fraud = red, legit = green.
        // The stage is darker than the rest of the app: the 3D has to sit on something
        // closer to black than the scoring tools do, or the bloom has nothing to bloom off.
        surface: { DEFAULT: "#111214", raised: "#181a1d", inset: "#0c0d0f", stage: "#06070a" },
        line: "#26292e",
        ink: { DEFAULT: "#f4f5f6", muted: "#a2a6ad", faint: "#6b6f77" },
        brand: { DEFAULT: "#3987e5", bright: "#6aa9ff", soft: "#1c3a5e" },
        fraud: { DEFAULT: "#e5484d", soft: "#3a1618" },
        legit: { DEFAULT: "#30a46c", soft: "#123024" },
        warn: { DEFAULT: "#f5b849", soft: "#3a2c10" },
        // The card's metal, reused for the two features the model actually leans on.
        gold: { DEFAULT: "#d9ae55", soft: "#3a2c10" },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.25)",
        "glow-brand": "0 0 0 1px rgba(57,135,229,0.4), 0 0 40px rgba(57,135,229,0.25)",
        "glow-fraud": "0 0 0 1px rgba(229,72,77,0.4), 0 0 40px rgba(229,72,77,0.25)",
      },
      keyframes: {
        aurora: {
          "0%, 100%": { transform: "translate(-10%, -10%) scale(1)" },
          "50%": { transform: "translate(10%, 5%) scale(1.15)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-8px)" },
        },
        "glow-pulse": {
          "0%, 100%": { opacity: "0.6" },
          "50%": { opacity: "1" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        draw: {
          "0%": { strokeDashoffset: "1" },
          "100%": { strokeDashoffset: "0" },
        },
      },
      animation: {
        aurora: "aurora 18s ease-in-out infinite",
        shimmer: "shimmer 3.5s linear infinite",
        float: "float 6s ease-in-out infinite",
        "glow-pulse": "glow-pulse 2.5s ease-in-out infinite",
        "fade-up": "fade-up 0.7s ease-out both",
      },
    },
  },
  plugins: [],
};
