"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns service status and version information.
    """
    return HealthResponse(
        status="healthy",
        service="infection-risk-api",
        version="0.1.0",
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes/container orchestration.
    
    Returns 200 if service is ready to accept traffic.
    """
    # TODO: Add checks for:
    # - Model loaded successfully
    # - Database connection if applicable
    # - Feature store connection
    
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes/container orchestration.
    
    Returns 200 if service is alive.
    """
    return {"status": "alive"}

