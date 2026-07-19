"""
backend/main.py
----------------
FastAPI application entrypoint.

Startup sequence:
  1. Structured logging configured
  2. Database tables created (idempotent)
  3. InferenceService loads 8 .joblib models + fits scalers (~10–30s)
  4. SimulationService starts background asyncio task (streams 4 engines)
  5. API routers mounted

Routes:
  /api/v1/infer/*       — Inference
  /api/v1/agent/*       — Agentic pipeline
  /api/v1/rag/*         — RAG pipeline
  /api/v1/workorders/*  — Work orders + HITL
  /api/v1/metrics       — Fleet metrics
  /ws/fleet             — WebSocket real-time feed
  /health               — Health check
  /docs                 — Swagger UI (auto-generated)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Path setup: ensure inference.* and dev.* are importable ──────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Also add NASA_CMAPSS_RUL_Project to path (needed by feature_builder → src.*)
_ML_ROOT = _REPO_ROOT / "NASA_CMAPSS_RUL_Project"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

# ── App imports ───────────────────────────────────────────────────────────────
from backend.core.config import settings
from backend.core.logging import get_logger, new_correlation_id, setup_logging
from backend.db.session import create_tables
from backend.services.inference_service import inference_service
from backend.services.simulation_service import simulation_service

# Routers
from backend.api.v1.routers import inference, agent, rag, workorders, metrics
from backend.api.v1.websocket import router as ws_router

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    # Startup
    setup_logging()
    logger.info("MechSage backend starting", version="1.0.0", env=settings.llm_provider)
    create_tables()
    logger.info("Database tables ready")

    await inference_service.startup()
    await simulation_service.startup()
    logger.info("All services ready — accepting requests")

    yield

    # Shutdown
    logger.info("MechSage backend shutting down")
    await simulation_service.shutdown()
    logger.info("Shutdown complete")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title="MechSage API",
    description=(
        "Production-grade predictive maintenance API. "
        "Serves RUL prediction, anomaly detection, RAG-powered diagnostics, "
        "and LangGraph agentic work-order drafting for Ironside Manufacturing assets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Inject a unique correlation ID into every request's log context."""
    cid = request.headers.get("X-Correlation-ID") or new_correlation_id()
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(inference.router, prefix="/api/v1")
app.include_router(agent.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(workorders.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(ws_router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], status_code=status.HTTP_200_OK)
async def health():
    """Liveness + readiness check."""
    return {
        "status": "ok",
        "inference_ready": inference_service.is_ready(),
        "simulation_running": simulation_service.is_running,
        "ws_connections": simulation_service.manager.connection_count(),
    }


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Check server logs."},
    )
