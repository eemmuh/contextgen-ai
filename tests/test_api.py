"""
API tests for the RAG-based Image Generation System.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.api.app import create_app


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a minimal JPEG file
            f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7),01444\x1f\x27=9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9")
            return f.name
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Image Model COCO API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_database_stats.return_value = {
                "total_images": 10,
                "total_embeddings": 20,
                "total_generations": 5
            }
            mock_db.return_value = mock_instance
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database"] == "connected"
            assert "stats" in data
    
    def test_list_images(self, client):
        """Test listing images."""
        response = client.get("/api/v1/images?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "images" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
    
    def test_upload_image(self, client, sample_image_file):
        """Test image upload."""
        with open(sample_image_file, "rb") as f:
            files = {"file": ("test.jpg", f, "image/jpeg")}
            data = {
                "description": "Test image",
                "tags": "test,sample"
            }
            response = client.post("/api/v1/images/upload", files=files, data=data)
        
        # Clean up
        os.unlink(sample_image_file)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Image uploaded successfully"
        assert "image_id" in data
        assert "file_path" in data
    
    def test_upload_invalid_file(self, client):
        """Test uploading invalid file type."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/v1/images/upload", files=files)
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    def test_search_images(self, client):
        """Test image search."""
        with patch('src.api.app.DatabaseEmbeddingManager') as mock_embedding:
            mock_instance = MagicMock()
            mock_instance.search_similar.return_value = [
                {
                    "image_id": 1,
                    "image_path": "test.jpg",
                    "similarity_score": 0.95
                }
            ]
            mock_embedding.return_value = mock_instance
            
            response = client.get("/api/v1/search?query=cat&k=5")
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "cat"
            assert "results" in data
            assert data["count"] == 1
    
    def test_generate_image(self, client):
        """Test image generation."""
        with patch('src.api.app.RAGManager') as mock_rag, \
             patch('src.api.app.ImageGenerator') as mock_generator, \
             patch('src.api.app.DatabaseManager') as mock_db:
            
            mock_rag_instance = MagicMock()
            mock_rag_instance.augment_prompt.return_value = "Enhanced prompt"
            mock_rag.return_value = mock_rag_instance
            
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate.return_value = "output/generated.jpg"
            mock_generator.return_value = mock_generator_instance
            
            mock_db_instance = MagicMock()
            mock_db_instance.add_generation.return_value = 1
            mock_db.return_value = mock_db_instance
            
            data = {
                "prompt": "a cat playing",
                "seed": 42,
                "num_inference_steps": 50,
                "guidance_scale": 7.5
            }
            response = client.post("/api/v1/generate", data=data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Image generated successfully"
            assert "generation_id" in data
            assert data["original_prompt"] == "a cat playing"
            assert data["augmented_prompt"] == "Enhanced prompt"
    
    def test_list_generations(self, client):
        """Test listing generations."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_generation_history.return_value = [
                {
                    "id": 1,
                    "prompt": "Test prompt",
                    "output_path": "output/test.jpg",
                    "created_at": "2024-01-01T00:00:00"
                }
            ]
            mock_db.return_value = mock_instance
            
            response = client.get("/api/v1/generations?limit=10")
            assert response.status_code == 200
            data = response.json()
            assert "generations" in data
            assert data["total"] == 1
            assert data["limit"] == 10
    
    def test_get_stats(self, client):
        """Test getting system statistics."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_database_stats.return_value = {
                "total_images": 100,
                "total_embeddings": 200,
                "total_generations": 50
            }
            mock_db.return_value = mock_instance
            
            response = client.get("/api/v1/stats")
            assert response.status_code == 200
            data = response.json()
            assert "database_stats" in data
            assert "timestamp" in data
            assert data["database_stats"]["total_images"] == 100
    
    def test_get_image_by_id(self, client):
        """Test getting image by ID."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_image = MagicMock()
            mock_image.id = 1
            mock_image.image_path = "test.jpg"
            mock_image.description = "Test image"
            mock_image.tags = ["test"]
            mock_image.width = 512
            mock_image.height = 512
            mock_image.created_at.isoformat.return_value = "2024-01-01T00:00:00"
            mock_instance.get_image_by_id.return_value = mock_image
            mock_db.return_value = mock_instance
            
            response = client.get("/api/v1/images/1")
            assert response.status_code == 200
            data = response.json()
            assert "image" in data
            assert data["image"]["id"] == 1
            assert data["image"]["image_path"] == "test.jpg"
    
    def test_get_image_not_found(self, client):
        """Test getting non-existent image."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_image_by_id.return_value = None
            mock_db.return_value = mock_instance
            
            response = client.get("/api/v1/images/999")
            assert response.status_code == 404
            assert "Image not found" in response.json()["detail"]
    
    def test_serve_file(self, client, sample_image_file):
        """Test serving static files."""
        # Create a test file in uploads directory
        os.makedirs("uploads", exist_ok=True)
        test_file_path = "uploads/test.jpg"
        with open(sample_image_file, "rb") as src, open(test_file_path, "wb") as dst:
            dst.write(src.read())
        
        try:
            response = client.get("/api/v1/files/test.jpg")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
        finally:
            # Clean up
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
            os.unlink(sample_image_file)
    
    def test_serve_file_not_found(self, client):
        """Test serving non-existent file."""
        response = client.get("/api/v1/files/nonexistent.jpg")
        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app = create_app()
        return TestClient(app)
    
    def test_database_connection_error(self, client):
        """Test handling of database connection errors."""
        with patch('src.api.app.DatabaseManager') as mock_db:
            mock_instance = MagicMock()
            mock_instance.get_database_stats.side_effect = Exception("Database connection failed")
            mock_db.return_value = mock_instance
            
            response = client.get("/health")
            assert response.status_code == 500
            assert "Service unhealthy" in response.json()["detail"]
    
    def test_search_error(self, client):
        """Test handling of search errors."""
        with patch('src.api.app.DatabaseEmbeddingManager') as mock_embedding:
            mock_instance = MagicMock()
            mock_instance.search_similar.side_effect = Exception("Search failed")
            mock_embedding.return_value = mock_instance
            
            response = client.get("/api/v1/search?query=test")
            assert response.status_code == 500
            assert "Failed to search images" in response.json()["detail"]
    
    def test_generation_error(self, client):
        """Test handling of generation errors."""
        with patch('src.api.app.RAGManager') as mock_rag:
            mock_instance = MagicMock()
            mock_instance.augment_prompt.side_effect = Exception("Generation failed")
            mock_rag.return_value = mock_instance
            
            data = {"prompt": "test prompt"}
            response = client.post("/api/v1/generate", data=data)
            assert response.status_code == 500
            assert "Failed to generate image" in response.json()["detail"] 