"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { classNames } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/about", label: "About" },
] as const;

export function TopNav() {
  const pathname = usePathname();
  return (
    <nav
      className="sticky top-0 z-30 border-b border-surface-border bg-surface-card/80 backdrop-blur"
      aria-label="Primary"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-6 py-2 text-sm">
        <div className="flex items-center gap-6">
          {LINKS.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={classNames(
                  "border-b-2 py-1 transition",
                  active
                    ? "border-brand-600 font-semibold text-brand-700 dark:text-brand-300"
                    : "border-transparent text-ink-muted hover:text-ink",
                )}
                aria-current={active ? "page" : undefined}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
        <ThemeToggle />
      </div>
    </nav>
  );
}
