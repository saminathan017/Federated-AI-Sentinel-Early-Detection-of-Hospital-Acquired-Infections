"""
FastAPI application for infection risk prediction.

Provides endpoints for scoring, explanations, and health checks.
Includes Prometheus metrics and request logging.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator

from src.serving.api.routers import explain, health, score
from src.serving.runtime.predictor import InfectionPredictor

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Global predictor instance
predictor: InfectionPredictor | None = None

# Prometheus metrics
prediction_counter = Counter(
    "infection_predictions_total",
    "Total number of infection risk predictions",
    ["outcome"],
)

prediction_latency = Histogram(
    "infection_prediction_latency_seconds",
    "Latency of infection predictions",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global predictor
    
    logger.info("Loading infection risk predictor...")
    
    from pathlib import Path
    
    model_path = Path(os.getenv("MODEL_CHECKPOINT_PATH", "models/temporal/best_model.pt"))
    
    try:
        predictor = InfectionPredictor(model_path=model_path)
        logger.info("Predictor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        predictor = None
    
    yield
    
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="Federated AI Sentinel",
    description="Privacy-preserving infection risk prediction API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
cors_origins = os.getenv("API_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    request_id = str(uuid.uuid4())
    
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path}",
        extra={"request_id": request_id},
    )
    
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    logger.info(
        f"Request {request_id}: completed in {duration:.3f}s with status {response.status_code}",
        extra={"request_id": request_id, "duration": duration, "status": response.status_code},
    )
    
    response.headers["X-Request-ID"] = request_id
    
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all exceptions and return structured error."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(score.router, prefix="/api/v1", tags=["prediction"])
app.include_router(explain.router, prefix="/api/v1", tags=["explainability"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Instrumentator for automatic metrics
Instrumentator().instrument(app).expose(app, endpoint="/api/v1/metrics")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Federated AI Sentinel API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def get_predictor() -> InfectionPredictor:
    """Dependency to get predictor instance."""
    if predictor is None:
        raise RuntimeError("Predictor not loaded. Check application startup logs.")
    return predictor


# Make predictor accessible to routers
app.state.get_predictor = get_predictor


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.serving.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )

