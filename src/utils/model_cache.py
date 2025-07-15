import os
import torch
from typing import Dict, Any, Optional, Union
import hashlib
import json
from pathlib import Path

class ModelCache:
    """
    A caching system for ML models to avoid repeated downloads and loading.
    Supports both in-memory and on-disk caching.
    """
    
    def __init__(self, cache_dir: str = ".model_cache"):
        """
        Initialize the model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for frequently used models
        self._memory_cache: Dict[str, Any] = {}
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        # Ensure all metadata values are JSON serializable
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(v) for v in obj]
            elif hasattr(obj, 'dtype') or (hasattr(obj, '__class__') and obj.__class__.__module__ == 'torch'):
                return str(obj)
            elif isinstance(obj, (bytes, bytearray)):
                return obj.decode(errors='replace')
            else:
                try:
                    json.dumps(obj)
                    return obj
                except Exception:
                    return str(obj)
        serializable_metadata = make_json_serializable(self.metadata)
        with open(self.metadata_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
    
    def _get_cache_key(self, model_type: str, model_name: str, **kwargs) -> str:
        """
        Generate a unique cache key for a model.
        
        Args:
            model_type: Type of model (e.g., 'clip', 'sentence_transformer', 'stable_diffusion')
            model_name: Name/ID of the model
            **kwargs: Additional parameters that affect the model
            
        Returns:
            Unique cache key
        """
        # Convert kwargs to JSON-serializable format
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, 'dtype'):  # Handle PyTorch dtypes
                serializable_kwargs[key] = str(value)
            elif hasattr(value, '__class__') and value.__class__.__module__ == 'torch':
                # Handle other PyTorch objects
                serializable_kwargs[key] = str(value)
            else:
                serializable_kwargs[key] = value
        
        # Create a hash of the parameters
        param_str = json.dumps(serializable_kwargs, sort_keys=True)
        combined = f"{model_type}:{model_name}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_cached_model(
        self,
        model_type: str,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached model if available.
        
        Args:
            model_type: Type of model
            model_name: Name/ID of the model
            device: Device to load the model on
            **kwargs: Additional parameters
            
        Returns:
            Cached model or None if not found
        """
        cache_key = self._get_cache_key(model_type, model_name, device=device, **kwargs)
        
        # Check in-memory cache first
        if cache_key in self._memory_cache:
            print(f"📦 Using in-memory cached {model_type} model: {model_name}")
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}"
        if cache_path.exists() and cache_key in self.metadata:
            try:
                print(f"💾 Loading cached {model_type} model from disk: {model_name}")
                model = self._load_model_from_disk(model_type, cache_path, device, **kwargs)
                if model is not None:
                    # Add to memory cache
                    self._memory_cache[cache_key] = model
                    return model
            except Exception as e:
                print(f"⚠️ Failed to load cached model: {e}")
                # Remove corrupted cache entry
                self._remove_cache_entry(cache_key)
        
        return None
    
    def cache_model(
        self,
        model: Any,
        model_type: str,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Cache a model in memory and optionally on disk.
        
        Args:
            model: The model to cache
            model_type: Type of model
            model_name: Name/ID of the model
            device: Device the model is on
            **kwargs: Additional parameters
        """
        cache_key = self._get_cache_key(model_type, model_name, device=device, **kwargs)
        
        # Cache in memory
        self._memory_cache[cache_key] = model
        
        # Cache on disk
        cache_path = self.cache_dir / f"{cache_key}"
        try:
            self._save_model_to_disk(model, model_type, cache_path, **kwargs)
            
            # Ensure all metadata values are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_json_serializable(v) for v in obj]
                elif hasattr(obj, 'dtype') or (hasattr(obj, '__class__') and obj.__class__.__module__ == 'torch'):
                    return str(obj)
                elif isinstance(obj, (bytes, bytearray)):
                    return obj.decode(errors='replace')
                else:
                    try:
                        json.dumps(obj)
                        return obj
                    except Exception:
                        return str(obj)
            
            serializable_kwargs = make_json_serializable(kwargs)
            
            self.metadata[cache_key] = {
                'model_type': model_type,
                'model_name': model_name,
                'device': device,
                'cache_path': str(cache_path),
                'parameters': serializable_kwargs,
                'created_at': str(torch.cuda.Event() if torch.cuda.is_available() else None)
            }
            self._save_metadata()
            
            print(f"💾 Cached {model_type} model to disk: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to cache model to disk: {e}")
    
    def _save_model_to_disk(self, model: Any, model_type: str, cache_path: Path, **kwargs):
        """Save a model to disk cache."""
        cache_path.mkdir(exist_ok=True)
        
        if model_type == "clip":
            # Import here to avoid circular imports
            from transformers import CLIPProcessor, CLIPModel
            # Save CLIP model and processor
            model.save_pretrained(cache_path / "model")
            if hasattr(model, 'processor'):
                model.processor.save_pretrained(cache_path / "processor")
        elif model_type == "sentence_transformer":
            # Save sentence transformer
            model.save(str(cache_path))
        elif model_type == "stable_diffusion":
            # Save Stable Diffusion pipeline
            model.save_pretrained(str(cache_path))
        else:
            # Generic torch save
            torch.save(model.state_dict(), cache_path / "model.pt")
    
    def _load_model_from_disk(self, model_type: str, cache_path: Path, device: str, **kwargs) -> Optional[Any]:
        """Load a model from disk cache."""
        try:
            if model_type == "clip":
                # Import here to avoid circular imports
                from transformers import CLIPProcessor, CLIPModel
                # Load CLIP model
                model = CLIPModel.from_pretrained(cache_path / "model")
                processor = CLIPProcessor.from_pretrained(cache_path / "processor")
                model.processor = processor
                return model.to(device)
            
            elif model_type == "sentence_transformer":
                # Import here to avoid circular imports
                from sentence_transformers import SentenceTransformer
                # Load sentence transformer
                return SentenceTransformer(str(cache_path), device=device)
            
            elif model_type == "stable_diffusion":
                # Import here to avoid circular imports
                from diffusers import StableDiffusionPipeline
                # Load Stable Diffusion pipeline
                torch_dtype = kwargs.get('torch_dtype', torch.float16 if device == "cuda" else torch.float32)
                pipeline = StableDiffusionPipeline.from_pretrained(
                    str(cache_path),
                    torch_dtype=torch_dtype
                )
                return pipeline.to(device)
            
            else:
                # Generic torch load
                model_state = torch.load(cache_path / "model.pt", map_location=device)
                # This would need the model class to be provided
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to load model from disk: {e}")
            return None
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry from both memory and disk."""
        # Remove from memory
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # Remove from disk
        if cache_key in self.metadata:
            cache_path = Path(self.metadata[cache_key]['cache_path'])
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
            del self.metadata[cache_key]
            self._save_metadata()
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear the cache.
        
        Args:
            model_type: If specified, only clear models of this type
        """
        if model_type is None:
            # Clear all cache
            self._memory_cache.clear()
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.metadata.clear()
            self._save_metadata()
            print("🗑️ Cleared all model cache")
        else:
            # Clear specific model type
            keys_to_remove = [
                key for key, info in self.metadata.items()
                if info['model_type'] == model_type
            ]
            for key in keys_to_remove:
                self._remove_cache_entry(key)
            print(f"🗑️ Cleared cache for {model_type} models")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        cache_size = sum(
            Path(info['cache_path']).stat().st_size 
            for info in self.metadata.values()
            if Path(info['cache_path']).exists()
        )
        
        return {
            'memory_cache_size': len(self._memory_cache),
            'disk_cache_size': len(self.metadata),
            'total_size_bytes': cache_size,
            'cached_models': {
                info['model_type']: info['model_name']
                for info in self.metadata.values()
            }
        }

# Global model cache instance
_model_cache = None

def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache 