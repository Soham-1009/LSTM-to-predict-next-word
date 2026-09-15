import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef2ff",
          100: "#e0e7ff",
          200: "#c7d2fe",
          300: "#a5b4fc",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
          800: "#3730a3",
          900: "#312e81",
        },
        // Semantic surface tokens used across the dashboard. Light values
        // are the defaults; dark variants are applied via the `dark:`
        // variant in component classes.
        surface: {
          page: "rgb(var(--surface-page) / <alpha-value>)",
          card: "rgb(var(--surface-card) / <alpha-value>)",
          muted: "rgb(var(--surface-muted) / <alpha-value>)",
          border: "rgb(var(--surface-border) / <alpha-value>)",
          ring: "rgb(var(--surface-ring) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--ink) / <alpha-value>)",
          muted: "rgb(var(--ink-muted) / <alpha-value>)",
          subtle: "rgb(var(--ink-subtle) / <alpha-value>)",
          inverse: "rgb(var(--ink-inverse) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
      },
      boxShadow: {
        soft: "0 4px 20px -2px rgba(15, 23, 42, 0.08)",
        "soft-dark": "0 4px 20px -2px rgba(0, 0, 0, 0.4)",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 240ms ease-out",
      },
    },
  },
  plugins: [],
};

export default config;
