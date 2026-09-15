export default function Loading() {
  return (
    <div
      className="flex min-h-screen items-center justify-center bg-surface-page text-ink-muted"
      role="status"
    >
      <div className="flex flex-col items-center gap-3">
        <span className="h-8 w-8 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
        <span className="text-sm font-medium">Loading dashboard…</span>
      </div>
    </div>
  );
}
