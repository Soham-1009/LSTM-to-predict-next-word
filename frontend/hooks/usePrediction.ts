"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type {
  GenerateResponse,
  GenerationStrategy,
  PredictResponse,
} from "@/types/api";

export type RequestStatus = "idle" | "loading" | "success" | "error";

export interface PredictionState {
  status: RequestStatus;
  error: string | null;
  predictions: PredictResponse | null;
  generated: GenerateResponse | null;
}

/**
 * usePrediction wraps both `/predict` and `/generate` in a small
 * state machine. The latest in-flight request is tracked so a new
 * click cancels the previous one.
 */
export function usePrediction() {
  const [state, setState] = useState<PredictionState>({
    status: "idle",
    error: null,
    predictions: null,
    generated: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState({ status: "idle", error: null, predictions: null, generated: null });
  }, []);

  const predict = useCallback(
    async (params: { text: string; topK: number }): Promise<PredictResponse | null> => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ status: "loading", error: null, predictions: null, generated: null });
      try {
        const res = await api.predictNextWord(params, controller.signal);
        setState({ status: "success", error: null, predictions: res, generated: null });
        return res;
      } catch (err) {
        if (controller.signal.aborted) return null;
        const message = err instanceof ApiError ? err.message : "Unexpected error";
        setState({ status: "error", error: message, predictions: null, generated: null });
        throw err;
      }
    },
    [],
  );

  const generate = useCallback(
    async (params: {
      text: string;
      numWords: number;
      temperature: number;
      strategy: GenerationStrategy;
    }): Promise<GenerateResponse | null> => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setState({ status: "loading", error: null, predictions: null, generated: null });
      try {
        const res = await api.generateText(params, controller.signal);
        setState({ status: "success", error: null, predictions: null, generated: res });
        return res;
      } catch (err) {
        if (controller.signal.aborted) return null;
        const message = err instanceof ApiError ? err.message : "Unexpected error";
        setState({ status: "error", error: message, predictions: null, generated: null });
        throw err;
      }
    },
    [],
  );

  useEffect(
    () => () => {
      abortRef.current?.abort();
    },
    [],
  );

  return { state, predict, generate, reset };
}
