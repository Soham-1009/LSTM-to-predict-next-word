"use client";

import React, { useEffect, useState } from "react";
import { fetchHealth, fetchModelInfo, HealthResponse, ModelInfoResponse } from "@/lib/api";

export default function ModelInfoBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    async function loadStatus() {
      try {
        const [h, m] = await Promise.all([fetchHealth(), fetchModelInfo()]);
        if (mounted) {
          setHealth(h);
          setModelInfo(m);
          setError(null);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Backend offline");
        }
      } finally {
        if (mounted) setLoading(false);
      }
    }
    loadStatus();
    const interval = setInterval(loadStatus, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-neutral-400 bg-slate-200/80 dark:bg-neutral-900/60 border border-slate-300 dark:border-neutral-800 rounded-full px-3 py-1">
        <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
        Connecting to FastAPI backend...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-800/50 rounded-full px-3 py-1">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        Backend offline (Run FastAPI on port 8000)
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-700 dark:text-neutral-300 bg-white/90 dark:bg-neutral-900/80 border border-slate-200/90 dark:border-neutral-800 rounded-lg px-3 py-1.5 shadow-sm">
      <div className="flex items-center gap-1.5">
        <span
          className={`w-2 h-2 rounded-full ${
            modelInfo?.checkpoint_loaded
              ? "bg-emerald-500 dark:bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
              : "bg-amber-500 dark:bg-amber-400"
          }`}
        />
        <span
          className={`font-medium ${
            modelInfo?.checkpoint_loaded ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"
          }`}
        >
          {modelInfo?.checkpoint_loaded ? "Online" : "Degraded"}
        </span>
      </div>
      <span className="text-slate-300 dark:text-neutral-600">|</span>
      <span>{modelInfo?.architecture ?? "LSTM"}</span>
      <span className="text-slate-300 dark:text-neutral-600">|</span>
      <span className="bg-slate-100 dark:bg-neutral-800 border border-slate-200 dark:border-transparent text-slate-800 dark:text-neutral-300 px-1.5 py-0.5 rounded font-mono uppercase text-[10px]">
        {health?.device ?? "CPU"}
      </span>
      <span className="text-slate-300 dark:text-neutral-600">|</span>
      <span className="text-slate-500 dark:text-neutral-400">Vocab: {modelInfo?.vocab_size.toLocaleString()}</span>
    </div>
  );
}
