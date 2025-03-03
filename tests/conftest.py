import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pytest and other test dependencies
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    """Create a test client fixture for FastAPI app."""
    return TestClient(app) 