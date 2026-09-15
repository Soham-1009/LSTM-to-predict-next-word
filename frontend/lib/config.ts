/**
 * Centralized frontend configuration. Every component reads the backend
 * URL from here, never from a hard-coded string.
 */

const DEFAULT_API_BASE_URL = "http://localhost:8000";

const raw =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ||
  DEFAULT_API_BASE_URL;

export const API_BASE_URL = raw;

export const REQUEST_TIMEOUT_MS = 30_000;
export const HISTORY_STORAGE_KEY = "lstm-prediction-history";
export const HISTORY_MAX_ENTRIES = 25;
export const BACKEND_STATUS_POLL_MS = 30_000;
