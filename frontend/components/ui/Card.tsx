import type { HTMLAttributes, ReactNode } from "react";
import { classNames } from "@/lib/utils";

interface CardProps extends HTMLAttributes<HTMLElement> {
  as?: "section" | "aside" | "article" | "div";
  children: ReactNode;
}

export function Card({ as = "section", className, children, ...rest }: CardProps) {
  const Tag = as;
  return (
    <Tag
      className={classNames(
        "rounded-2xl border border-surface-border bg-surface-card p-6 text-ink shadow-soft transition-colors dark:shadow-soft-dark",
        className,
      )}
      {...rest}
    >
      {children}
    </Tag>
  );
}
