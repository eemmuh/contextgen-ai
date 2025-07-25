"""
Integration tests for database functionality.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Image, Embedding, Generation
from src.database.database import DatabaseManager
from src.database.session import get_db_session


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture
    def db_manager(self):
        """Create a database manager for testing."""
        return DatabaseManager()

    @pytest.fixture
    def sample_embedding(self):
        """Create a sample embedding for testing."""
        return np.random.rand(384).astype(np.float32)

    def test_add_image(self, db_manager):
        """Test adding an image to the database."""
        image_path = "test_images/test.jpg"
        image_id = db_manager.add_image(
            image_path=image_path, description="Test image", tags=["test", "sample"], width=512, height=512
        )

        assert image_id is not None
        assert isinstance(image_id, int)

        # Verify image was added
        image = db_manager.get_image_by_path(image_path)
        assert image is not None
        assert image.image_path == image_path
        assert image.description == "Test image"

    def test_add_embedding(self, db_manager, sample_embedding):
        """Test adding an embedding to the database."""
        # First add an image
        image_id = db_manager.add_image("test_images/test.jpg")

        # Add embedding
        embedding_id = db_manager.add_embedding(
            embedding=sample_embedding,
            image_id=image_id,
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            embedding_type="text",
        )

        assert embedding_id is not None
        assert isinstance(embedding_id, int)

    def test_search_similar_embeddings(self, db_manager, sample_embedding):
        """Test vector similarity search."""
        # Add test data
        image_id = db_manager.add_image("test_images/test.jpg")
        db_manager.add_embedding(
            embedding=sample_embedding,
            image_id=image_id,
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            embedding_type="text",
        )

        # Search for similar embeddings
        results = db_manager.search_similar_embeddings(
            query_embedding=sample_embedding, model_type="sentence_transformer", embedding_type="text", k=5
        )

        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("similarity_score" in r for r in results)

    def test_add_generation(self, db_manager):
        """Test adding a generation record."""
        generation_id = db_manager.add_generation(
            prompt="Test prompt", augmented_prompt="Augmented test prompt", output_path="output/test.jpg", seed=42
        )

        assert generation_id is not None
        assert isinstance(generation_id, int)

    def test_get_generation_history(self, db_manager):
        """Test retrieving generation history."""
        # Add some test generations
        db_manager.add_generation("Test prompt 1", "Augmented 1", "output/1.jpg")
        db_manager.add_generation("Test prompt 2", "Augmented 2", "output/2.jpg")

        history = db_manager.get_generation_history(limit=5)

        assert len(history) >= 2
        assert all(isinstance(h, dict) for h in history)
        assert all("prompt" in h for h in history)

    def test_database_stats(self, db_manager):
        """Test database statistics."""
        stats = db_manager.get_database_stats()

        assert isinstance(stats, dict)
        assert "total_images" in stats
        assert "total_embeddings" in stats
        assert "total_generations" in stats
        assert all(isinstance(v, int) for v in stats.values())


class TestDatabaseSession:
    """Tests for database session management."""

    def test_session_context_manager(self):
        """Test session context manager."""
        with get_db_session() as session:
            assert session is not None
            # Test basic query
            result = session.execute("SELECT 1").scalar()
            assert result == 1

    def test_session_error_handling(self):
        """Test session error handling."""
        with pytest.raises(Exception):
            with get_db_session() as session:
                # This should raise an error
                session.execute("SELECT * FROM non_existent_table")


class TestDatabaseModels:
    """Tests for database models."""

    def test_image_model(self):
        """Test Image model creation."""
        image = Image(image_path="test.jpg", description="Test image", tags=["test"], width=512, height=512)

        assert image.image_path == "test.jpg"
        assert image.description == "Test image"
        assert image.tags == ["test"]

    def test_embedding_model(self):
        """Test Embedding model creation."""
        embedding = Embedding(
            image_id=1,
            embedding=np.random.rand(384).tolist(),
            model_type="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            embedding_type="text",
        )

        assert embedding.image_id == 1
        assert embedding.model_type == "sentence_transformer"
        assert embedding.embedding_type == "text"

    def test_generation_model(self):
        """Test Generation model creation."""
        generation = Generation(
            prompt="Test prompt", augmented_prompt="Augmented prompt", output_path="output/test.jpg", seed=42
        )

        assert generation.prompt == "Test prompt"
        assert generation.augmented_prompt == "Augmented prompt"
        assert generation.seed == 42
