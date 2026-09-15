"use client";

import { useTheme } from "@/hooks/useTheme";
import { classNames } from "@/lib/utils";

interface ThemeToggleProps {
  className?: string;
}

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { theme, toggle, ready } = useTheme();

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      aria-pressed={theme === "dark"}
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className={classNames(
        "inline-flex h-9 w-9 items-center justify-center rounded-full border border-surface-border bg-surface-card text-ink-muted shadow-soft transition hover:bg-surface-muted hover:text-ink focus-visible:ring-2 focus-visible:ring-brand-300",
        className,
      )}
    >
      <span aria-hidden="true" className="text-base">
        {ready ? (theme === "dark" ? "☀" : "☾") : "◐"}
      </span>
    </button>
  );
}
