"use client";

import { useCallback, useEffect, useState } from "react";
import { HISTORY_MAX_ENTRIES, HISTORY_STORAGE_KEY } from "@/lib/config";
import type { HistoryEntry } from "@/types/api";

function isHistoryEntry(value: unknown): value is HistoryEntry {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.id === "string" &&
    typeof v.ts === "number" &&
    (v.kind === "predict" || v.kind === "generate") &&
    typeof v.input === "string" &&
    typeof v.output === "string"
  );
}

function readInitial(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isHistoryEntry);
  } catch {
    return [];
  }
}

/**
 * Lightweight, localStorage-backed history of recent predictions. No
 * external state library, no backend, no database. Guarded against
 * private-mode and quota errors so a failure here never breaks the UI.
 */
export function useLocalHistory() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setHistory(readInitial());
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // Ignore quota / private-mode errors.
    }
  }, [history]);

  const addEntry = useCallback((entry: Omit<HistoryEntry, "id" | "ts">) => {
    setHistory((prev) => {
      const created: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        ts: Date.now(),
        ...entry,
      };
      return [created, ...prev].slice(0, HISTORY_MAX_ENTRIES);
    });
  }, []);

  const removeEntry = useCallback((id: string) => {
    setHistory((prev) => prev.filter((entry) => entry.id !== id));
  }, []);

  const clear = useCallback(() => setHistory([]), []);

  return { history, addEntry, removeEntry, clear };
}
