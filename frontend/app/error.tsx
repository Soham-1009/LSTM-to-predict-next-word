"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/Button";

interface ErrorPageProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: ErrorPageProps) {
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.error("Dashboard error boundary caught:", error);
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-page p-6 text-ink">
      <div className="max-w-md rounded-2xl border border-rose-200 bg-surface-card p-8 text-center shadow-soft dark:border-rose-500/40">
        <h1 className="text-2xl font-semibold">Something went wrong</h1>
        <p className="mt-2 text-sm text-ink-muted">
          The dashboard ran into an unexpected error. Your input and history are safe.
        </p>
        <div className="mt-6 flex justify-center gap-2">
          <Button type="button" onClick={() => reset()}>
            Try again
          </Button>
          <Button
            type="button"
            variant="secondary"
            onClick={() => {
              if (typeof window !== "undefined") window.location.reload();
            }}
          >
            Reload
          </Button>
        </div>
      </div>
    </div>
  );
}
