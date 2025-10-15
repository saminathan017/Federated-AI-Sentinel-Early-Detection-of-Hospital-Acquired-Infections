"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

# Note: This assumes the API can run without a model loaded (for testing)
# In practice, you might want to mock the predictor


def test_health_check():
    """Test health check endpoint."""
    from src.serving.api.main import app
    
    client = TestClient(app)
    
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check():
    """Test readiness endpoint."""
    from src.serving.api.main import app
    
    client = TestClient(app)
    
    response = client.get("/api/v1/ready")
    
    assert response.status_code == 200
    assert "status" in response.json()


def test_liveness_check():
    """Test liveness endpoint."""
    from src.serving.api.main import app
    
    client = TestClient(app)
    
    response = client.get("/api/v1/live")
    
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_root_endpoint():
    """Test root endpoint."""
    from src.serving.api.main import app
    
    client = TestClient(app)
    
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["name"] == "Federated AI Sentinel API"
    assert "docs" in data


# Note: Testing score and explain endpoints requires a loaded model
# These would typically use fixtures or mocks in a full test suite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

