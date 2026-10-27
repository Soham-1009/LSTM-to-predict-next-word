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
        setConnMessage({ text: "Connected successfully to backend", ok: true });
        setTimeout(() => setShowConfig(false), 2000);
      } else {
        setConnMessage({ text: "Backend unreachable at this URL", ok: false });
      }
    } catch {
      setConnMessage({ text: "Connection failed. Check URL and CORS.", ok: false });
    } finally {
      setTestingConnection(false);
    }
  };

  const isOnline = health?.status === "ok" && modelInfo?.checkpoint_loaded;
  const isDegraded = health?.status === "ok" && !modelInfo?.checkpoint_loaded;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setShowConfig(!showConfig)}
        title="View or configure backend connection settings"
        className="flex items-center gap-2 text-[13px] text-[var(--muted)] hover:text-[var(--text)] bg-[var(--surface)] hover:bg-[var(--raised)] border border-[var(--border)] rounded px-2.5 py-1.5 transition-colors duration-150 cursor-pointer"
      >
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{
            backgroundColor: isOnline
              ? "var(--positive)"
              : isDegraded
              ? "var(--warning)"
              : "var(--accent)",
          }}
        />
        <span className="font-medium text-[var(--text)]">
          {isOnline ? "Online" : isDegraded ? "Degraded" : "Static Demo"}
        </span>
        <span className="text-[var(--border)]">/</span>
        <span className="hidden sm:inline text-[var(--muted)]">
          {modelInfo?.architecture ? "2-Layer LSTM" : "LSTM"}
        </span>
        <span className="text-[var(--border)] hidden sm:inline">/</span>
        <span className="tabular-nums text-[var(--muted)]">
          {health?.device ?? "CPU"}
        </span>
        <span className="text-[var(--border)] hidden md:inline">/</span>
        <span className="tabular-nums text-[var(--muted)] hidden md:inline">
          30,000 vocab
        </span>
      </button>

      {/* Backend Configuration Modal / Popover */}
      {showConfig && (
        <div className="absolute right-0 top-full mt-2 w-80 sm:w-96 bg-[var(--surface)] border border-[var(--border)] rounded p-4 z-50 text-[13px]">
          <div className="flex items-center justify-between pb-2.5 mb-3 border-b border-[var(--rule)]">
            <h3 className="font-semibold text-sm text-[var(--text)]">
              Backend Connection
            </h3>
            <button
              onClick={() => setShowConfig(false)}
              className="p-1 text-[var(--muted)] hover:text-[var(--text)] rounded cursor-pointer"
              aria-label="Close connection settings"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <p className="text-[var(--muted)] mb-1">
                <strong>Status:</strong>{" "}
                <span className="text-[var(--text)]">
                  {isOnline
                    ? "Connected to active FastAPI inference backend"
                    : isDegraded
                    ? "Connected, but PyTorch checkpoint weights missing"
                    : "Static Showcase (Bundled multi-domain benchmark & specs)"}
                </span>
              </p>
              <p className="text-xs text-[var(--faint)] leading-relaxed">
                Static deployment hosts offline evaluation metrics and architectural specs. Connect to FastAPI for real-time inference.
              </p>
            </div>

            <div className="bg-[var(--ground)] p-3 rounded border border-[var(--border)] space-y-2">
              <label className="block text-xs font-medium text-[var(--muted)]">
                FastAPI Backend URL:
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customUrl}
                  onChange={(e) => setCustomUrl(e.target.value)}
                  placeholder="http://127.0.0.1:8000"
                  className="flex-1 px-2.5 py-1.5 text-xs bg-[var(--surface)] border border-[var(--control)] rounded text-[var(--text)] focus:outline-none focus:border-[var(--accent)] font-mono"
                />
                <button
                  type="button"
                  onClick={handleSaveUrl}
                  disabled={testingConnection}
                  className="px-3 py-1.5 bg-[var(--accent)] hover:opacity-90 text-white rounded font-medium text-xs transition-opacity cursor-pointer disabled:opacity-50"
                >
                  {testingConnection ? "..." : "Connect"}
                </button>
              </div>
              {connMessage && (
                <p
                  className="text-xs"
                  style={{ color: connMessage.ok ? "var(--positive)" : "var(--negative)" }}
                >
                  {connMessage.text}
                </p>
              )}
            </div>

            <div className="text-xs text-[var(--muted)] space-y-1">
              <p className="font-medium text-[var(--text)]">Run inference locally:</p>
              <code className="block bg-[var(--ground)] p-2 rounded text-xs text-[var(--text)] border border-[var(--rule)]">
                uvicorn backend.app.main:app --port 8000
              </code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
