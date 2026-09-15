import type { Metadata } from "next";
import Link from "next/link";
import { Card } from "@/components/ui/Card";

export const metadata: Metadata = {
  title: "About · LSTM Next Word Predictor",
  description:
    "How the LSTM Next Word Predictor works: architecture, training, and API integration.",
};

const STACK = [
  { label: "Frontend", value: "Next.js 14 (App Router) + TypeScript + Tailwind CSS" },
  { label: "Backend", value: "FastAPI + Pydantic" },
  { label: "ML Framework", value: "PyTorch 2.x" },
  { label: "Model", value: "Embedding → LSTM (1–3 layers) → Dropout → Linear → Softmax" },
  { label: "Inference", value: "Top-k softmax probabilities and autoregressive generation" },
] as const;

const ENDPOINTS = [
  { method: "GET", path: "/", purpose: "Service banner" },
  { method: "GET", path: "/health", purpose: "Health + model/device status" },
  { method: "GET", path: "/model-info", purpose: "Architecture & vocabulary metadata" },
  { method: "POST", path: "/predict", purpose: "Top-k next-word predictions" },
  { method: "POST", path: "/generate", purpose: "Autoregressive text generation" },
] as const;

export default function AboutPage() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
      <Link
        href="/"
        className="text-sm font-medium text-brand-700 transition hover:text-brand-800 dark:text-brand-300 dark:hover:text-brand-200"
      >
        ← Back to dashboard
      </Link>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight text-ink">
        About this project
      </h1>
      <p className="mt-2 text-ink-muted">
        A Generative AI college project that performs next-word prediction and text
        generation using a PyTorch LSTM served by a FastAPI backend.
      </p>

      <Card className="mt-8">
        <h2 className="text-lg font-semibold text-ink">Stack</h2>
        <dl className="mt-4 grid grid-cols-1 gap-y-3 sm:grid-cols-3 sm:gap-x-6">
          {STACK.map(({ label, value }) => (
            <div key={label} className="contents">
              <dt className="text-sm font-medium text-ink-muted">{label}</dt>
              <dd className="col-span-2 text-sm text-ink">{value}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card className="mt-6">
        <h2 className="text-lg font-semibold text-ink">API endpoints</h2>
        <p className="mt-1 text-sm text-ink-muted">
          All requests go to <code className="font-mono text-ink">NEXT_PUBLIC_API_BASE_URL</code>.
        </p>
        <table className="mt-4 w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-ink-muted">
              <th className="pb-2">Method</th>
              <th className="pb-2">Path</th>
              <th className="pb-2">Purpose</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-surface-border">
            {ENDPOINTS.map((ep) => (
              <tr key={ep.path}>
                <td className="py-2 pr-3 font-mono text-brand-700 dark:text-brand-300">
                  {ep.method}
                </td>
                <td className="py-2 pr-3 font-mono text-ink">{ep.path}</td>
                <td className="py-2 text-ink-muted">{ep.purpose}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </main>
  );
}
