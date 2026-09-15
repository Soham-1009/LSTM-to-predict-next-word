import Link from "next/link";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-page p-6 text-ink">
      <div className="max-w-md rounded-2xl border border-surface-border bg-surface-card p-8 text-center shadow-soft">
        <h1 className="text-3xl font-semibold">404</h1>
        <p className="mt-2 text-sm text-ink-muted">
          The page you were looking for doesn&apos;t exist.
        </p>
        <div className="mt-6 flex justify-center">
          <Link href="/">
            <Button type="button">Back to dashboard</Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
