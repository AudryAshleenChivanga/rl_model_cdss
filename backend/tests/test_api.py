"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api.main import app


class TestAPI:
    """Test suite for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "H. pylori" in response.text

    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "disclaimer" in data

    def test_metrics(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "gpu_available" in data

    def test_disclaimer(self, client):
        """Test disclaimer endpoint."""
        response = client.get("/api/disclaimer")
        assert response.status_code == 200
        
        data = response.json()
        assert "title" in data
        assert "warnings" in data
        assert len(data["warnings"]) > 0

    def test_models_info(self, client):
        """Test models info endpoint."""
        response = client.get("/api/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "cnn" in data
        assert "ppo" in data

    def test_load_model_invalid_url(self, client):
        """Test loading model with invalid URL."""
        response = client.post("/api/load_model?gltf_url=invalid_url")
        assert response.status_code == 400

    def test_sim_status_not_running(self, client):
        """Test simulation status when not running."""
        response = client.get("/api/sim/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["running"] == False

    def test_stop_sim_not_running(self, client):
        """Test stopping simulation when not running."""
        response = client.post("/api/sim/stop")
        assert response.status_code == 400


def test_api_smoke():
    """Quick smoke test."""
    client = TestClient(app)
    
    # Test health
    response = client.get("/api/health")
    assert response.status_code == 200
    
    # Test metrics
    response = client.get("/api/metrics")
    assert response.status_code == 200
    
    print("API smoke test passed!")


if __name__ == "__main__":
    test_api_smoke()

