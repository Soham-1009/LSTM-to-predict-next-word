# Frontend — LSTM Next Word Predictor (Next.js)

This directory is the **Next.js 14 (App Router) + TypeScript + Tailwind CSS**
frontend for the LSTM Next Word Predictor project. It was migrated from an
earlier React + Vite implementation; the FastAPI backend is unchanged.

```
Browser
   ↓
Next.js (this app)         ← http://localhost:3000
   ↓   HTTP/JSON
FastAPI backend            ← http://localhost:8000
   ↓
PyTorch LSTM
```

The frontend never executes the model. Every prediction, generation, and
model-info response is produced by the FastAPI backend.

## Prerequisites

| Tool | Version |
| ---- | ------- |
| Node.js | 18.17+ (or 20+) |
| npm | 9+ (or pnpm / yarn) |
| Backend | FastAPI on `localhost:8000` (see `../backend/README.md`) |

> The frontend is pure Node.js — there is **no Python venv** in this
> directory. All Python setup lives in `../backend/`.

## Install & run

```powershell
cd C:\Users\soham\Desktop\Soham\GenAI\frontend
npm install
copy .env.example .env.local
npm run dev
```

Dashboard: <http://localhost:3000>

## Environment variables

Set these in `.env.local` (gitignored):

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Where the FastAPI backend lives. Must be reachable from the browser. |

`NEXT_PUBLIC_*` variables are baked into the browser bundle at build time.

## Scripts

```powershell
npm run dev      # next dev (hot reload)
npm run build    # production build
npm run start    # next start (serve the production build)
npm run lint     # eslint
npm run test     # vitest unit tests
```

## Project structure

```
frontend/
├── app/                            ← App Router
│   ├── layout.tsx                  ← metadata, <html>/<body>, TopNav
│   ├── page.tsx                    ← / (dashboard)
│   ├── about/page.tsx              ← /about (architecture & API reference)
│   ├── loading.tsx                 ← global loading UI
│   ├── error.tsx                   ← global error boundary
│   ├── not-found.tsx               ← 404 page
│   └── globals.css                 ← Tailwind + base styles
│
├── components/
│   ├── dashboard/                  ← top-level layout
│   │   ├── Dashboard.tsx           ← the client component that composes everything
│   │   ├── Header.tsx
│   │   └── TopNav.tsx
│   ├── prediction/                 ← next-word prediction
│   │   ├── PredictionPanel.tsx
│   │   └── PredictionResult.tsx
│   ├── generation/                 ← text generation
│   │   ├── GenerationPanel.tsx
│   │   └── GenerationResult.tsx
│   ├── history/                    ← localStorage history
│   │   └── HistoryPanel.tsx
│   ├── model/                      ← architecture / metadata
│   │   └── ModelInfoPanel.tsx
│   ├── education/                  ← "How it works" explainer
│   │   └── HowItWorksPanel.tsx
│   ├── ui/                         ← primitives
│   │   ├── Button.tsx
│   │   └── Card.tsx
│   └── __tests__/                  ← vitest + RTL component tests
│
├── hooks/                          ← React hooks
│   ├── useBackendStatus.ts
│   ├── usePrediction.ts
│   ├── useLocalHistory.ts
│   └── __tests__/
│
├── lib/                            ← framework-agnostic utilities
│   ├── api.ts                      ← typed FastAPI client (only place fetch is used)
│   ├── config.ts                   ← API_BASE_URL, storage keys, timeouts
│   ├── utils.ts                    ← formatPercent, formatTime, truncate, classNames
│   └── __tests__/
│
├── types/                          ← TypeScript mirrors of Pydantic schemas
│   └── api.ts
│
├── public/
│   └── favicon.svg
│
├── package.json
├── tsconfig.json
├── next.config.ts
├── tailwind.config.ts
├── postcss.config.mjs
├── vitest.config.ts
├── vitest.setup.ts
├── .eslintrc.json
├── .env.example
└── README.md
```

## How requests flow

1. The user types in `PredictionPanel` or `GenerationPanel`.
2. The component calls a hook (`usePrediction`) that delegates to
   `lib/api.ts`.
3. `lib/api.ts` issues a typed `fetch` to the FastAPI backend.
4. The response is parsed, validated against the TypeScript types, and
   returned to the hook.
5. The hook updates state, and the result component re-renders.

There is no model inference in the browser. The only place `fetch` is
called is `lib/api.ts`.

## API contract

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/` | Service banner |
| `GET`  | `/health` | Health, model loaded, device |
| `GET`  | `/model-info` | Architecture & vocabulary metadata |
| `POST` | `/predict` | Top-k next-word predictions |
| `POST` | `/generate` | Autoregressive text generation |

Request and response shapes live in `types/api.ts` and mirror the
Pydantic schemas in `../backend/app/schemas.py`. Update both ends
together.

## CORS

The FastAPI backend's default `ALLOWED_ORIGINS` includes
`http://localhost:3000`, so the Next.js dev server works out of the
box. If you change the port, update the backend env accordingly.

## Testing

```powershell
npm run test
```

Tests use `vitest` + `@testing-library/react` + `jsdom`. Network
calls are mocked, so the suite runs in seconds without a live
backend.

## Build & deploy

```powershell
npm run build
npm run start
```

`next build` performs TypeScript checking, ESLint, and the production
bundling. The output is in `.next/`.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| Dashboard shows "Backend offline" | Make sure FastAPI is running on the URL in `NEXT_PUBLIC_API_BASE_URL`. |
| `CORS` errors in the browser console | Add your frontend origin to `ALLOWED_ORIGINS` in `backend/.env` and restart FastAPI. |
| `npm run build` fails on type errors | The dashboard uses strict TypeScript. `npx tsc --noEmit` to inspect. |
| `vitest` complains about JSX | `vitest.setup.ts` imports `@testing-library/jest-dom`. Make sure it is present. |
| "I have Python 3.14" | The frontend has nothing to do with Python. Make sure you're running `npm` commands in PowerShell, and the Python venv (if you also need the backend) is created from a **3.10 or 3.11** interpreter — see `../README.md` for the steps. |
