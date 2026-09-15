"""FastAPI application entry point.

Uses the lifespan context manager to load the model exactly once at startup
and avoid any per-request loading overhead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .api.routes import router
from .services.model_loader import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lstm_backend")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model on startup. Log clearly if artifacts are missing."""
    try:
        model_manager.load()
        logger.info("LSTM model ready for inference.")
    except FileNotFoundError as exc:
        logger.warning("Model artifacts missing: %s", exc)
        logger.warning(
            "The API will start but /predict, /generate, and /model-info "
            "will return 503 until artifacts are produced by training/train.py"
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unexpected error while loading model: %s", exc)
    yield
    # No teardown needed: model is in-memory only.


app = FastAPI(
    title=config.APP_TITLE,
    description=config.APP_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS -- restricted to configured origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic errors to the spec'd `{ "detail": ... }` shape."""
    # Pull a clean message instead of dumping the full error list.
    first_error = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first_error.get("loc", []))
    msg = first_error.get("msg", "Invalid request.")
    human = f"{field}: {msg}" if field else msg
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": human},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler. Never leaks a stack trace to the client."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


app.include_router(router)
