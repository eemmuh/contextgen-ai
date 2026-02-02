"""
Database-backed embedding manager for the RAG-based Image Generation System.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import os
from src.utils.model_cache import get_model_cache
from src.database.database import DatabaseManager
from src.utils.logger import get_logger

logger = get_logger("database_embedding_manager")


class DatabaseEmbeddingManager:
    """
    Database-backed embedding manager that stores embeddings in PostgreSQL with pgvector.
    """

    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 384,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        database_manager: Optional[DatabaseManager] = None,
    ):
        """
        Initialize the database-backed embedding manager.

        Args:
            text_model_name: Name of the sentence transformer model
            image_model_name: Name of the CLIP model
            embedding_dim: Dimension of the embeddings
            device: Device to run the models on
            database_manager: Optional database manager instance
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.database_manager = database_manager or DatabaseManager()

        # Get model cache
        self.model_cache = get_model_cache()

        # Initialize models with caching
        logger.info(f"Loading text model: {text_model_name}")
        self.text_model = self._load_text_model(text_model_name)

        logger.info(f"Loading image model: {image_model_name}")
        self.image_model, self.image_processor = self._load_image_model(image_model_name)

    def _load_text_model(self, model_name: str) -> SentenceTransformer:
        """Load text model with caching."""
        # Try to get from cache first
        cached_model = self.model_cache.get_cached_model(
            model_type="sentence_transformer", model_name=model_name, device=self.device
        )

        if cached_model is not None:
            return cached_model

        # Load from HuggingFace and cache
        logger.info(f"Downloading text model: {model_name}")
        model = SentenceTransformer(model_name, device=self.device)

        # Cache the model
        self.model_cache.cache_model(
            model=model,
            model_type="sentence_transformer",
            model_name=model_name,
            device=self.device,
        )

        return model

    def _load_image_model(self, model_name: str) -> tuple[CLIPModel, CLIPProcessor]:
        """Load image model with caching."""
        # Try to get from cache first
        cached_model = self.model_cache.get_cached_model(model_type="clip", model_name=model_name, device=self.device)

        if cached_model is not None:
            # For CLIP, we need to handle processor separately
            processor = CLIPProcessor.from_pretrained(model_name)
            return cached_model, processor

        # Load from HuggingFace and cache
        logger.info(f"Downloading image model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(model_name)

        # Attach processor to model for caching
        model.processor = processor

        # Cache the model
        self.model_cache.cache_model(model=model, model_type="clip", model_name=model_name, device=self.device)

        return model, processor

    def compute_text_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text input."""
        with torch.no_grad():
            embedding = self.text_model.encode(text, convert_to_tensor=True)
            return embedding.cpu().numpy()

    def compute_image_embedding(self, image: torch.Tensor) -> np.ndarray:
        """Compute embedding for an image input."""
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            image_features = self.image_model.get_image_features(image)
            # Project CLIP's 512-dim features to match text embedding dimension
            image_features = torch.nn.functional.linear(
                image_features,
                torch.randn(self.embedding_dim, image_features.shape[-1], device=self.device),
            )
            return image_features.cpu().numpy()

    def add_image_with_embeddings(
        self,
        image_path: str,
        metadata: Dict[str, Any],
        compute_embeddings: bool = True,
        embedding_types: List[str] = None,
    ) -> int:
        """
        Add an image to the database with optional embeddings.

        Args:
            image_path: Path to the image file
            metadata: Image metadata dictionary
            compute_embeddings: Whether to compute embeddings
            embedding_types: Types of embeddings to compute ('text', 'image', or both)

        Returns:
            Image ID in the database
        """
        # Add image to database (extra fields stored in image_metadata)
        image_id = self.database_manager.add_image(
            image_path=image_path,
            description=metadata.get("description"),
            tags=metadata.get("tags", []),
            width=metadata.get("width"),
            height=metadata.get("height"),
            metadata={
                **{k: v for k, v in metadata.items() if k not in ("description", "tags", "width", "height")},
            },
        )

        if compute_embeddings:
            embedding_types = embedding_types or ["text"]

            for embedding_type in embedding_types:
                if embedding_type == "text":
                    # Compute text embedding from metadata
                    metadata_text = self._format_metadata_text(metadata)
                    text_embedding = self.compute_text_embedding(metadata_text)

                    self.database_manager.add_embedding(
                        image_id=image_id,
                        embedding=text_embedding,
                        model_type="sentence_transformer",
                        model_name="all-MiniLM-L6-v2",
                        embedding_type="text",
                        metadata={"source_text": metadata_text},
                    )

                elif embedding_type == "image":
                    # Load and process the actual image
                    try:
                        from PIL import Image
                        import torchvision.transforms as transforms
                        
                        # Load image
                        image = Image.open(image_path).convert('RGB')
                        
                        # Preprocess for CLIP model
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225])
                        ])
                        
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        # Compute image embedding
                        image_embedding = self.compute_image_embedding(image_tensor)
                        
                        self.database_manager.add_embedding(
                            image_id=image_id,
                            embedding=image_embedding,
                            model_type="clip",
                            model_name=self.image_model_name,
                            embedding_type="image",
                            metadata={"image_path": image_path, "image_size": image.size},
                        )
                        
                        logger.info(f"Computed image embedding for {image_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to compute image embedding for {image_path}: {e}")
                        # Continue with other embedding types even if image embedding fails

        return image_id

    def search_similar(
        self,
        query: str,
        model_type: str = "sentence_transformer",
        embedding_type: str = "text",
        k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using text query.

        Args:
            query: Text query to search for
            model_type: Type of model to use for embedding
            embedding_type: Type of embedding to search
            k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar images with metadata and similarity scores
        """
        # Compute query embedding
        query_embedding = self.compute_text_embedding(query)

        # Search in database
        results = self.database_manager.search_similar_embeddings(
            query_embedding=query_embedding,
            model_type=model_type,
            embedding_type=embedding_type,
            k=k,
            similarity_threshold=similarity_threshold,
        )

        # Enrich results with image metadata using direct SQL to avoid session issues
        enriched_results = []
        for result in results:
            # Get image metadata directly from database
            with self.database_manager.get_db_session_context() as session:
                from sqlalchemy import text

                image_result = session.execute(
                    text("SELECT image_path, description, tags FROM images WHERE id = :image_id"),
                    {"image_id": result["image_id"]},
                ).fetchone()

                if image_result:
                    enriched_results.append(
                        {
                            "image_id": result["image_id"],
                            "image_path": image_result.image_path,
                            "description": image_result.description,
                            "tags": image_result.tags,
                            "similarity_score": result["similarity_score"],
                            "metadata": result["metadata"],
                        }
                    )

        return enriched_results

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.database_manager.get_database_stats()

    def _format_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Format metadata into searchable text."""
        parts = []

        if metadata.get("description"):
            parts.append(metadata["description"])

        if metadata.get("tags"):
            parts.extend(metadata["tags"])

        if metadata.get("width") and metadata.get("height"):
            parts.append(f"Size: {metadata['width']}x{metadata['height']}")

        return " | ".join(parts) if parts else "No description available"

    def migrate_from_faiss(self, faiss_index_path: str, metadata_path: str, batch_size: int = 100) -> int:
        """
        Migrate data from FAISS index to database.

        Args:
            faiss_index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            batch_size: Number of items to process in each batch

        Returns:
            Number of items migrated
        """
        import faiss
        import pickle

        try:
            # Load FAISS index and metadata
            index = faiss.read_index(faiss_index_path)
            with open(metadata_path, "rb") as f:
                metadata_store = pickle.load(f)

            logger.info(f"Migrating {len(metadata_store)} items from FAISS to database...")

            migrated_count = 0

            for i, item in enumerate(metadata_store):
                try:
                    # Get embedding from FAISS index
                    embedding = index.reconstruct(i)

                    # Add image to database
                    image_id = self.add_image_with_embeddings(
                        image_path=item["image_path"],
                        metadata=item["metadata"],
                        compute_embeddings=False,  # We'll add the existing embedding
                    )

                    # Add the existing embedding
                    self.database_manager.add_embedding(
                        image_id=image_id,
                        embedding=embedding,
                        model_type="sentence_transformer",
                        model_name="all-MiniLM-L6-v2",
                        embedding_type="text",
                        metadata={"migrated_from_faiss": True, "original_index": i},
                    )

                    migrated_count += 1

                    if (i + 1) % batch_size == 0:
                        logger.info(f"Migrated {i + 1}/{len(metadata_store)} items")

                except Exception as e:
                    logger.error(f"Failed to migrate item {i}: {e}")
                    continue

            logger.info(f"Migration completed: {migrated_count}/{len(metadata_store)} items migrated")
            return migrated_count

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
