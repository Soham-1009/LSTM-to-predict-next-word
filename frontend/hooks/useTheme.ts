"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark";
const STORAGE_KEY = "lstm-theme";

function readInitial(): Theme {
  if (typeof window === "undefined") return "light";
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* private mode / quota */
  }
  if (window.matchMedia?.("(prefers-color-scheme: dark)").matches) {
    return "dark";
  }
  return "light";
}

function apply(theme: Theme): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (theme === "dark") {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
  root.style.colorScheme = theme;
}

/**
 * useTheme exposes the current theme plus a toggle.  The initial value
 * is computed on the client to avoid an SSR/CSR mismatch.
 */
export function useTheme(): { theme: Theme; toggle: () => void; setTheme: (next: Theme) => void; ready: boolean } {
  const [theme, setThemeState] = useState<Theme>("light");
  const [ready, setReady] = useState(false);

  // Hydrate from storage / system preference once on mount.
  useEffect(() => {
    const initial = readInitial();
    setThemeState(initial);
    apply(initial);
    setReady(true);
  }, []);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    apply(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  }, []);

  const toggle = useCallback(() => {
    setTheme(theme === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  return { theme, toggle, setTheme, ready };
}
