"""Integration tests for FastAPI."""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'status' in data
        assert 'version' in data
        assert 'components' in data
        assert 'uptime_seconds' in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert 'text/plain' in response.headers['content-type']
        assert 'rag_' in response.text
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'service' in data
        assert 'version' in data
        assert 'endpoints' in data
    
    @pytest.mark.slow
    def test_query_endpoint_valid(self, client):
        """Test query endpoint with valid request."""
        payload = {
            "query": "Is the staff friendly?",
            "top_k": 5,
            "enable_filtering": True,
            "return_contexts": False
        }
        
        response = client.post("/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'query' in data
        assert 'answer' in data
        assert 'citations' in data
        assert 'metrics' in data
        assert isinstance(data['citations'], list)
    
    def test_query_endpoint_invalid(self, client):
        """Test query endpoint with invalid request."""
        payload = {
            "query": "",  # Empty query
            "top_k": 5
        }
        
        response = client.post("/query", json=payload)
        
        assert response.status_code == 422  # Validation error

