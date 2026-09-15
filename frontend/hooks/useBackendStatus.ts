"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { BACKEND_STATUS_POLL_MS } from "@/lib/config";
import type { HealthResponse, HealthState } from "@/types/api";

export interface BackendStatus {
  state: HealthState;
  data: HealthResponse | null;
  error: string | null;
}

/**
 * Polls `/health` on mount and on a configurable interval. Does not
 * infer backend availability from any prior success — every check
 * actually hits the network.
 */
export function useBackendStatus(pollMs: number = BACKEND_STATUS_POLL_MS): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>({
    state: "checking",
    data: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;

    async function check(): Promise<void> {
      try {
        const data = await api.healthCheck();
        if (cancelled) return;
        setStatus({
          state: data.model_loaded ? "online" : "degraded",
          data,
          error: null,
        });
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof ApiError ? err.message : "Backend unavailable";
        setStatus({ state: "offline", data: null, error: message });
      }
    }

    void check();
    const id = setInterval(() => void check(), pollMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollMs]);

  return status;
}
