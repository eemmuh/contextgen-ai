"""
Pytest configuration and fixtures for the Image Model COCO project.

This module provides common fixtures and configuration for all tests.
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Generator, Dict, Any

from config.settings import get_settings, reload_settings


@pytest.fixture(scope="session")
def test_settings() -> Dict[str, Any]:
    """Get test settings."""
    # Set test environment
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DB_URL"] = "postgresql://postgres:postgres@localhost:5433/image_rag_db_test"
    os.environ["MODEL_DEVICE"] = "cpu"
    os.environ["CACHE_ENABLED"] = "false"

    # Reload settings with test environment
    settings = reload_settings()
    return settings


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def test_data_dir(temp_dir: Path) -> Path:
    """Create test data directory."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def test_logs_dir(temp_dir: Path) -> Path:
    """Create test logs directory."""
    logs_dir = temp_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


@pytest.fixture(scope="session")
def test_models_dir(temp_dir: Path) -> Path:
    """Create test models directory."""
    models_dir = temp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@pytest.fixture(scope="function")
def mock_settings(test_settings: Dict[str, Any], temp_dir: Path) -> Dict[str, Any]:
    """Mock settings for individual tests."""
    # Update paths to use temporary directories
    test_settings["data_dir"] = temp_dir / "data"
    test_settings["logs_dir"] = temp_dir / "logs"
    test_settings["models_dir"] = temp_dir / "models"

    return test_settings


@pytest.fixture(scope="function")
def sample_image_path(test_data_dir: Path) -> Path:
    """Create a sample image file for testing."""
    image_path = test_data_dir / "sample_image.jpg"

    # Create a simple test image (1x1 pixel JPEG)
    with open(image_path, "wb") as f:
        # Minimal JPEG file content
        f.write(
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7),01444\x1f\x27=9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9"
        )

    return image_path


@pytest.fixture(scope="function")
def sample_text_data() -> str:
    """Sample text data for testing."""
    return "A cat sitting on a chair in a sunny room"


@pytest.fixture(scope="function")
def sample_embedding_vector() -> list:
    """Sample embedding vector for testing."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 100-dimensional vector


@pytest.fixture(scope="function")
def mock_database_url() -> str:
    """Mock database URL for testing."""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture(scope="function")
def mock_api_response() -> Dict[str, Any]:
    """Mock API response for testing."""
    return {"status": "success", "data": {"id": "test_id", "result": "test_result"}, "message": "Test successful"}


@pytest.fixture(scope="function")
def mock_error_response() -> Dict[str, Any]:
    """Mock error response for testing."""
    return {"status": "error", "error": {"code": "TEST_ERROR", "message": "Test error message"}}


# Database fixtures
@pytest.fixture(scope="function")
def db_session():
    """Database session fixture."""
    # This would be implemented with actual database connection
    # For now, return None to indicate no database session
    return None


@pytest.fixture(scope="function")
def sample_image_record() -> Dict[str, Any]:
    """Sample image database record."""
    return {
        "id": 1,
        "file_path": "/path/to/image.jpg",
        "file_name": "test_image.jpg",
        "file_size": 1024,
        "width": 512,
        "height": 512,
        "format": "JPEG",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture(scope="function")
def sample_embedding_record() -> Dict[str, Any]:
    """Sample embedding database record."""
    return {
        "id": 1,
        "image_id": 1,
        "embedding_type": "text",
        "embedding_vector": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
        "model_name": "test-model",
        "created_at": "2024-01-01T00:00:00Z",
    }


# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_test_data() -> Dict[str, Any]:
    """Data for performance testing."""
    return {"batch_size": 100, "num_iterations": 10, "timeout_seconds": 30}


# Cleanup fixtures
@pytest.fixture(scope="function", autouse=True)
def cleanup_test_files(temp_dir: Path):
    """Clean up test files after each test."""
    yield
    # Cleanup is handled by the temp_dir fixture


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "database: marks tests that require database")
    config.addinivalue_line("markers", "api: marks tests that test API endpoints")
