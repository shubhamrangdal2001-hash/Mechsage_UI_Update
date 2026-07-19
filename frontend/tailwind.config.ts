// frontend/tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base:     "#0A0F1E",
          surface:  "#111827",
          elevated: "#1F2937",
          overlay:  "#0D1424",
        },
        accent: {
          teal:   "#00D4AA",
          amber:  "#F59E0B",
          red:    "#EF4444",
          blue:   "#3B82F6",
          orange: "#F97316",
        },
        text: {
          primary:   "#F9FAFB",
          secondary: "#D1D5DB",
          muted:     "#6B7280",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        "xl":  "12px",
        "2xl": "16px",
      },
      animation: {
        "fade-in":  "fade-in 0.3s ease-out",
        "slide-in": "slide-in-right 0.3s ease-out",
        "pulse-emergency": "pulse-emergency 2s infinite",
        "pulse-critical":  "pulse-critical 2s infinite",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          from: { opacity: "0", transform: "translateX(20px)" },
          to:   { opacity: "1", transform: "translateX(0)" },
        },
        "pulse-emergency": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(239,68,68,0.4)" },
          "50%":       { boxShadow: "0 0 0 8px rgba(239,68,68,0)" },
        },
        "pulse-critical": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(245,158,11,0.4)" },
          "50%":       { boxShadow: "0 0 0 8px rgba(245,158,11,0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
