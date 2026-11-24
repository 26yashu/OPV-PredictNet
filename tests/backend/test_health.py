import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import settings
from backend.app.main import app, get_health


class TestHealthEndpoint(unittest.TestCase):
    """Test suite for the GET /api/health endpoint."""

    def test_get_health_direct(self):
        """Verify the health handler returns valid status and metadata."""
        response = get_health()
        self.assertIsInstance(response, dict)
        self.assertEqual(response["status"], "healthy")
        self.assertEqual(response["service"], "OPV-PredictNet Backend")
        self.assertIn("version", response)
        self.assertIn("timestamp", response)
        self.assertIn("environment", response)

    def test_health_route_registered(self):
        """Verify that /api/health is registered on the FastAPI app."""
        route_paths = list(app.openapi()["paths"].keys())
        self.assertIn("/api/health", route_paths)

    def test_cors_middleware_configured(self):
        """Verify CORS middleware is present in the FastAPI application stack."""
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        self.assertIn("CORSMiddleware", middleware_names)

    def test_cors_origins_configured(self):
        """Verify CORS allowed origins include Vite dev server origins."""
        self.assertIn("http://localhost:5173", settings.CORS_ORIGINS)
        self.assertIn("http://127.0.0.1:5173", settings.CORS_ORIGINS)


if __name__ == "__main__":
    unittest.main()
