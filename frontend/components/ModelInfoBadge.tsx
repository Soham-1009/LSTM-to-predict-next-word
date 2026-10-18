"use client";

import React, { useEffect, useState } from "react";
import {
  fetchHealth,
  fetchModelInfo,
  getApiBaseUrl,
  setApiBaseUrl,
  HealthResponse,
  ModelInfoResponse,
} from "@/lib/api";

export default function ModelInfoBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [customUrl, setCustomUrl] = useState("");
  const [testingConnection, setTestingConnection] = useState(false);
  const [connMessage, setConnMessage] = useState<{ text: string; ok: boolean } | null>(null);

  useEffect(() => {
    setCustomUrl(getApiBaseUrl());
  }, []);

  const loadStatus = async () => {
    try {
      const [h, m] = await Promise.all([fetchHealth(), fetchModelInfo()]);
      setHealth(h);
      setModelInfo(m);
    } catch {
      setHealth({ status: "offline", version: "0.1.0", device: "CPU" });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;
    loadStatus();
    const interval = setInterval(() => {
      if (mounted) loadStatus();
    }, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleSaveUrl = async () => {
    setTestingConnection(true);
    setConnMessage(null);
    setApiBaseUrl(customUrl);
    try {
      const [h, m] = await Promise.all([fetchHealth(), fetchModelInfo()]);
      setHealth(h);
      setModelInfo(m);
      if (h.status === "ok") {
        setConnMessage({ text: "Connected successfully to backend!", ok: true });
        setTimeout(() => setShowConfig(false), 2000);
      } else {
        setConnMessage({ text: "Backend unreachable at this URL.", ok: false });
      }
    } catch {
      setConnMessage({ text: "Connection failed. Check URL and CORS.", ok: false });
    } finally {
      setTestingConnection(false);
    }
  };

  const isOnline = health?.status === "ok" && modelInfo?.checkpoint_loaded;
  const isDegraded = health?.status === "ok" && !modelInfo?.checkpoint_loaded;
  const isStaticDemo = health?.status === "offline";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setShowConfig(!showConfig)}
        title="Click to view or configure backend connection"
        className="flex flex-wrap items-center gap-2 text-xs text-slate-700 dark:text-neutral-300 bg-white/90 dark:bg-neutral-900/80 hover:bg-slate-50 dark:hover:bg-neutral-800 border border-slate-200/90 dark:border-neutral-800 rounded-lg px-3 py-1.5 shadow-sm transition cursor-pointer"
      >
        <div className="flex items-center gap-1.5">
          <span
            className={`w-2 h-2 rounded-full ${
              isOnline
                ? "bg-emerald-500 dark:bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]"
                : isDegraded
                ? "bg-amber-500 dark:bg-amber-400"
                : "bg-sky-500 dark:bg-sky-400 animate-pulse"
            }`}
          />
          <span
            className={`font-medium ${
              isOnline
                ? "text-emerald-600 dark:text-emerald-400"
                : isDegraded
                ? "text-amber-600 dark:text-amber-400"
                : "text-sky-600 dark:text-sky-400"
            }`}
          >
            {isOnline ? "Online" : isDegraded ? "Degraded" : "Static Demo"}
          </span>
        </div>
        <span className="text-slate-300 dark:text-neutral-600">|</span>
        <span className="hidden sm:inline">{modelInfo?.architecture ?? "LSTM"}</span>
        <span className="text-slate-300 dark:text-neutral-600 hidden sm:inline">|</span>
        <span className="bg-slate-100 dark:bg-neutral-800 border border-slate-200 dark:border-transparent text-slate-800 dark:text-neutral-300 px-1.5 py-0.5 rounded font-mono uppercase text-[10px]">
          {health?.device ?? "CPU"}
        </span>
        <span className="text-slate-300 dark:text-neutral-600">|</span>
        <span className="text-slate-500 dark:text-neutral-400">Vocab: 30,000</span>
      </button>

      {/* Backend Configuration Modal / Popover */}
      {showConfig && (
        <div className="absolute right-0 top-full mt-2 w-80 sm:w-96 bg-white dark:bg-neutral-900 border border-slate-200 dark:border-neutral-800 rounded-xl shadow-xl p-4 z-50 text-xs">
          <div className="flex items-center justify-between pb-2 mb-3 border-b border-slate-100 dark:border-neutral-800">
            <h3 className="font-semibold text-slate-900 dark:text-white flex items-center gap-1.5">
              <span>Backend Connection Settings</span>
            </h3>
            <button
              onClick={() => setShowConfig(false)}
              className="text-slate-400 hover:text-slate-600 dark:hover:text-neutral-200"
            >
              ✕
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <p className="text-slate-600 dark:text-neutral-400 mb-1">
                <strong>Current State:</strong>{" "}
                <span className={isOnline ? "text-emerald-500" : "text-sky-400"}>
                  {isOnline
                    ? "Connected to active FastAPI inference server"
                    : isDegraded
                    ? "Connected, but PyTorch model weights missing"
                    : "Static GitHub Pages Showcase (Bundled Evaluation & Specs)"}
                </span>
              </p>
              <p className="text-[11px] text-slate-500 dark:text-neutral-400 leading-relaxed">
                GitHub Pages hosts the static Next.js frontend. The Multi-Domain Evaluation dashboard and architecture specs are fully functional offline.
              </p>
            </div>

            <div className="bg-slate-50 dark:bg-neutral-950 p-2.5 rounded-lg border border-slate-200 dark:border-neutral-800">
              <label className="block text-[11px] font-medium text-slate-700 dark:text-neutral-300 mb-1">
                FastAPI Backend URL:
              </label>
              <div className="flex gap-1.5">
                <input
                  type="text"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                  placeholder="http://127.0.0.1:8000"
                  className="flex-1 px-2.5 py-1 text-xs bg-white dark:bg-neutral-900 border border-slate-300 dark:border-neutral-700 rounded text-slate-900 dark:text-white font-mono focus:outline-none focus:ring-1 focus:ring-cyan-500"
                />
                <button
                  type="button"
                  onClick={handleSaveUrl}
                  disabled={testingConnection}
                  className="px-3 py-1 bg-cyan-600 hover:bg-cyan-500 text-white rounded font-medium text-xs transition disabled:opacity-50"
                >
                  {testingConnection ? "..." : "Connect"}
                </button>
              </div>
              {connMessage && (
                <p className={`mt-1.5 text-[11px] ${connMessage.ok ? "text-emerald-500" : "text-red-400"}`}>
                  {connMessage.text}
                </p>
              )}
            </div>

            <div className="text-[11px] text-slate-500 dark:text-neutral-400">
              <p className="font-semibold text-slate-700 dark:text-neutral-300 mb-0.5">To run live inference locally:</p>
              <code className="block bg-slate-100 dark:bg-neutral-950 px-2 py-1 rounded font-mono text-[10px] text-slate-800 dark:text-neutral-300">
                uvicorn backend.app.main:app --port 8000
              </code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
