import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
from src.utils.model_cache import get_model_cache

class EmbeddingManager:
    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        embedding_dim: int = 384,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the embedding manager for handling both text and image embeddings.
        
        Args:
            text_model_name: Name of the sentence transformer model
            image_model_name: Name of the CLIP model
            embedding_dim: Dimension of the embeddings
            device: Device to run the models on
        """
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Get model cache
        self.model_cache = get_model_cache()
        
        # Initialize models with caching
        print(f"🔄 Loading text model: {text_model_name}")
        self.text_model = self._load_text_model(text_model_name)
        
        print(f"🔄 Loading image model: {image_model_name}")
        self.image_model, self.image_processor = self._load_image_model(image_model_name)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata_store = []
    
    def _load_text_model(self, model_name: str) -> SentenceTransformer:
        """Load text model with caching."""
        # Try to get from cache first
        cached_model = self.model_cache.get_cached_model(
            model_type="sentence_transformer",
            model_name=model_name,
            device=self.device
        )
        
        if cached_model is not None:
            return cached_model
        
        # Load from HuggingFace and cache
        print(f"📥 Downloading text model: {model_name}")
        model = SentenceTransformer(model_name, device=self.device)
        
        # Cache the model
        self.model_cache.cache_model(
            model=model,
            model_type="sentence_transformer",
            model_name=model_name,
            device=self.device
        )
        
        return model
    
    def _load_image_model(self, model_name: str) -> tuple[CLIPModel, CLIPProcessor]:
        """Load image model with caching."""
        # Try to get from cache first
        cached_model = self.model_cache.get_cached_model(
            model_type="clip",
            model_name=model_name,
            device=self.device
        )
        
        if cached_model is not None:
            # For CLIP, we need to handle processor separately
            processor = CLIPProcessor.from_pretrained(model_name)
            return cached_model, processor
        
        # Load from HuggingFace and cache
        print(f"📥 Downloading image model: {model_name}")
        model = CLIPModel.from_pretrained(model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # Attach processor to model for caching
        model.processor = processor
        
        # Cache the model
        self.model_cache.cache_model(
            model=model,
            model_type="clip",
            model_name=model_name,
            device=self.device
        )
        
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
                torch.randn(self.embedding_dim, image_features.shape[-1], device=self.device)
            )
            return image_features.cpu().numpy()
    
    def add_to_index(
        self,
        embedding: np.ndarray,
        metadata: Dict,
        image_path: Optional[str] = None
    ) -> int:
        """
        Add an embedding to the FAISS index along with its metadata.
        
        Returns:
            Index of the added embedding
        """
        embedding = embedding.reshape(1, -1)
        self.index.add(embedding)
        self.metadata_store.append({
            'metadata': metadata,
            'image_path': image_path
        })
        return len(self.metadata_store) - 1
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for similar items in the index.
        
        Args:
            query_embedding: Query embedding to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing metadata and similarity scores
        """
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid index
                results.append({
                    'metadata': self.metadata_store[idx]['metadata'],
                    'image_path': self.metadata_store[idx]['image_path'],
                    'similarity_score': float(1 / (1 + dist))  # Convert distance to similarity
                })
        
        return results
    
    def save_index(self, path: str):
        """Save the FAISS index and metadata store."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(self.metadata_store, f)
    
    def load_index(self, path: str):
        """Load the FAISS index and metadata store."""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.metadata", 'rb') as f:
            self.metadata_store = pickle.load(f) 