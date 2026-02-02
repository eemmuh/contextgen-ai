import torch
from typing import Dict, Any, Optional, List
import hashlib
import json
import time
import shutil
from pathlib import Path
from collections import OrderedDict
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cache entry with metadata and access tracking."""

    def __init__(
        self,
        model: Any,
        model_type: str,
        model_name: str,
        device: str,
        parameters: Dict,
        cache_path: Path,
    ):
        self.model = model
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.parameters = parameters
        self.cache_path = cache_path
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.size_bytes = 0

    def access(self):
        """Mark this entry as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1


class ModelCache:
    """
    An advanced caching system for ML models with size limits, eviction policies,
    compression, and monitoring capabilities.
    """

    def __init__(
        self,
        cache_dir: str = ".model_cache",
        max_memory_size_mb: int = 2048,  # 2GB memory limit
        max_disk_size_mb: int = 10240,  # 10GB disk limit
        compression_enabled: bool = True,
        enable_validation: bool = True,
    ):
        """
        Initialize the advanced model cache.

        Args:
            cache_dir: Directory to store cached models
            max_memory_size_mb: Maximum memory cache size in MB
            max_disk_size_mb: Maximum disk cache size in MB
            compression_enabled: Whether to compress cached models
            enable_validation: Whether to validate cached models
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Cache limits
        self.max_memory_size = max_memory_size_mb * 1024 * 1024  # Convert to bytes
        self.max_disk_size = max_disk_size_mb * 1024 * 1024
        self.compression_enabled = compression_enabled
        self.enable_validation = enable_validation

        # In-memory cache with LRU ordering
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_cache_size = 0

        # Cache metadata and statistics
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.stats_file = self.cache_dir / "cache_stats.json"
        self._load_metadata()
        self._load_stats()

        # Thread safety
        self._lock = threading.RLock()

        # Cache warming and background tasks
        self._warmup_queue = []
        self._background_thread = None

        logger.info(
            f"ModelCache initialized: memory_limit={max_memory_size_mb}MB, "
            f"disk_limit={max_disk_size_mb}MB, compression={compression_enabled}"
        )

    def _load_metadata(self):
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            serializable_metadata = self._make_json_serializable(self.metadata)
            with open(self.metadata_file, "w") as f:
                json.dump(serializable_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _load_stats(self):
        """Load cache statistics from disk."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, "r") as f:
                    self.stats = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")
                self.stats = self._get_default_stats()
        else:
            self.stats = self._get_default_stats()

    def _save_stats(self):
        """Save cache statistics to disk."""
        try:
            with open(self.stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save stats: {e}")

    def _get_default_stats(self) -> Dict[str, Any]:
        """Get default statistics structure."""
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "compression_savings": 0,
            "validation_failures": 0,
            "last_reset": time.time(),
        }

    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(v) for v in obj]
        elif hasattr(obj, "dtype") or (hasattr(obj, "__class__") and obj.__class__.__module__ == "torch"):
            return str(obj)
        elif isinstance(obj, (bytes, bytearray)):
            return obj.decode(errors="replace")
        else:
            try:
                json.dumps(obj)
                return obj
            except Exception:
                return str(obj)

    def _get_cache_key(self, model_type: str, model_name: str, **kwargs) -> str:
        """Generate a unique cache key for a model."""
        serializable_kwargs = {}
        for key, value in kwargs.items():
            if hasattr(value, "dtype"):
                serializable_kwargs[key] = str(value)
            elif hasattr(value, "__class__") and value.__class__.__module__ == "torch":
                serializable_kwargs[key] = str(value)
            else:
                serializable_kwargs[key] = value

        param_str = json.dumps(serializable_kwargs, sort_keys=True)
        combined = f"{model_type}:{model_name}:{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _estimate_model_size(self, model: Any) -> int:
        """Estimate the size of a model in bytes."""
        try:
            if hasattr(model, "state_dict"):
                # PyTorch model
                total_params = sum(p.numel() * p.element_size() for p in model.parameters())
                return total_params
            elif hasattr(model, "get_sentence_embedding_dimension"):
                # SentenceTransformer
                return model.get_sentence_embedding_dimension() * 4 * 1024  # Rough estimate
            else:
                # Generic estimate
                return 100 * 1024 * 1024  # 100MB default
        except Exception:
            return 100 * 1024 * 1024  # 100MB default

    def _evict_if_needed(self, required_size: int = 0):
        """Evict entries if cache limits are exceeded."""
        with self._lock:
            # Check memory cache
            while self._memory_cache_size + required_size > self.max_memory_size and self._memory_cache:
                # Remove least recently used entry
                key, entry = self._memory_cache.popitem(last=False)
                self._memory_cache_size -= entry.size_bytes
                self.stats["evictions"] += 1
                logger.info(f"Evicted from memory cache: {entry.model_name}")

            # Check disk cache size
            current_disk_size = self._get_disk_cache_size()
            if current_disk_size + required_size > self.max_disk_size:
                self._evict_disk_cache(required_size)

    def _evict_disk_cache(self, required_size: int):
        """Evict entries from disk cache based on LRU policy."""
        # Sort entries by last accessed time
        entries = []
        for cache_key, metadata in self.metadata.items():
            if "last_accessed" in metadata:
                entries.append(
                    (
                        cache_key,
                        metadata["last_accessed"],
                        metadata.get("size_bytes", 0),
                    )
                )

        entries.sort(key=lambda x: x[1])  # Sort by last accessed time

        freed_space = 0
        for cache_key, _, size in entries:
            if freed_space >= required_size:
                break

            try:
                cache_path = self.cache_dir / cache_key
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    freed_space += size
                    del self.metadata[cache_key]
                    self.stats["evictions"] += 1
                    logger.info(f"Evicted from disk cache: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to evict {cache_key}: {e}")

        self._save_metadata()

    def _get_disk_cache_size(self) -> int:
        """Calculate current disk cache size."""
        total_size = 0
        for cache_key in self.metadata:
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                try:
                    total_size += sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                except Exception:
                    pass
        return total_size

    def _validate_cached_model(self, model: Any, model_type: str) -> bool:
        """Validate a cached model."""
        if not self.enable_validation:
            return True

        try:
            if model_type == "clip":
                # Test CLIP model with dummy input
                if hasattr(model, "get_text_features"):
                    dummy_input = torch.randint(0, 1000, (1, 10))
                    _ = model.get_text_features(dummy_input)
                    return True
            elif model_type == "sentence_transformer":
                # Test sentence transformer
                if hasattr(model, "encode"):
                    _ = model.encode("test", convert_to_tensor=True)
                    return True
            elif model_type == "stable_diffusion":
                # Test Stable Diffusion pipeline
                if hasattr(model, "scheduler"):
                    return True

            return True
        except Exception as e:
            logger.warning(f"Model validation failed: {e}")
            self.stats["validation_failures"] += 1
            return False

    def get_cached_model(
        self,
        model_type: str,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> Optional[Any]:
        """Get a cached model if available."""
        with self._lock:
            cache_key = self._get_cache_key(model_type, model_name, device=device, **kwargs)
            self.stats["total_requests"] += 1

            # Check in-memory cache first
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                entry.access()
                # Move to end (most recently used)
                self._memory_cache.move_to_end(cache_key)
                self.stats["cache_hits"] += 1
                logger.info(f"Memory cache hit: {model_name}")
                return entry.model

            # Check disk cache
            cache_path = self.cache_dir / cache_key
            if cache_path.exists() and cache_key in self.metadata:
                try:
                    logger.info(f"Loading from disk cache: {model_name}")
                    model = self._load_model_from_disk(model_type, cache_path, device, **kwargs)

                    if model is not None and self._validate_cached_model(model, model_type):
                        # Create cache entry
                        entry = CacheEntry(
                            model=model,
                            model_type=model_type,
                            model_name=model_name,
                            device=device,
                            parameters=kwargs,
                            cache_path=cache_path,
                        )
                        entry.size_bytes = self._estimate_model_size(model)

                        # Add to memory cache (with eviction if needed). Skip if it exceeds the in-memory limit.
                        if entry.size_bytes <= self.max_memory_size:
                            self._evict_if_needed(entry.size_bytes)
                            self._memory_cache[cache_key] = entry
                            self._memory_cache_size += entry.size_bytes

                        # Update metadata
                        self.metadata[cache_key]["last_accessed"] = time.time()
                        self.metadata[cache_key]["access_count"] = self.metadata[cache_key].get("access_count", 0) + 1
                        self._save_metadata()

                        self.stats["cache_hits"] += 1
                        return model
                    else:
                        logger.warning(f"Model validation failed for {model_name}")
                        self._remove_cache_entry(cache_key)

                except Exception as e:
                    logger.error(f"Failed to load cached model: {e}")
                    self._remove_cache_entry(cache_key)

            self.stats["cache_misses"] += 1
            return None

    def cache_model(
        self,
        model: Any,
        model_type: str,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        """Cache a model in memory and on disk."""
        with self._lock:
            cache_key = self._get_cache_key(model_type, model_name, device=device, **kwargs)

            # Create cache entry
            entry = CacheEntry(
                model=model,
                model_type=model_type,
                model_name=model_name,
                device=device,
                parameters=kwargs,
                cache_path=self.cache_dir / cache_key,
            )
            entry.size_bytes = self._estimate_model_size(model)

            # Add to memory cache (with eviction if needed).
            # If a single model exceeds the cache limit, do not keep it in the in-memory cache.
            cache_in_memory = entry.size_bytes <= self.max_memory_size
            if cache_in_memory:
                self._evict_if_needed(entry.size_bytes)
                self._memory_cache[cache_key] = entry
                self._memory_cache_size += entry.size_bytes
            else:
                logger.info(
                    f"Skipping in-memory cache for {model_name}: "
                    f"model_size_mb={entry.size_bytes / (1024 * 1024):.1f} exceeds "
                    f"memory_limit_mb={self.max_memory_size / (1024 * 1024):.1f}"
                )

            # Cache on disk
            try:
                self._save_model_to_disk(model, model_type, entry.cache_path, **kwargs)

                # Update metadata
                self.metadata[cache_key] = {
                    "model_type": model_type,
                    "model_name": model_name,
                    "device": device,
                    "cache_path": str(entry.cache_path),
                    "parameters": self._make_json_serializable(kwargs),
                    "created_at": time.time(),
                    "last_accessed": time.time(),
                    "access_count": 1,
                    "size_bytes": entry.size_bytes,
                }
                self._save_metadata()

                logger.info(f"Cached {model_type} model: {model_name}")

            except Exception as e:
                logger.error(f"Failed to cache model to disk: {e}")

    def _save_model_to_disk(self, model: Any, model_type: str, cache_path: Path, **kwargs):
        """Save a model to disk cache with optional compression."""
        cache_path.mkdir(exist_ok=True)

        if model_type == "clip":
            # from transformers import CLIPProcessor, CLIPModel (imported locally where needed)
            model.save_pretrained(cache_path / "model")
            if hasattr(model, "processor"):
                model.processor.save_pretrained(cache_path / "processor")
        elif model_type == "sentence_transformer":
            model.save(str(cache_path))
        elif model_type == "stable_diffusion":
            model.save_pretrained(str(cache_path))
        else:
            # Generic torch model: attempt to persist state_dict.
            state_dict = None
            if hasattr(model, "state_dict") and callable(getattr(model, "state_dict")):
                try:
                    state_dict = model.state_dict()
                except TypeError:
                    # Some tests use a mock "state_dict" without a "self" parameter; fall back to the class attr.
                    try:
                        state_dict_fn = getattr(type(model), "state_dict", None)
                        if callable(state_dict_fn):
                            state_dict = state_dict_fn()
                    except Exception:
                        state_dict = None

            if state_dict is None:
                raise ValueError("Model does not provide a usable state_dict for disk caching")

            torch.save(state_dict, cache_path / "model.pt")

        # Apply compression if enabled
        if self.compression_enabled:
            self._compress_cache_directory(cache_path)

    def _compress_cache_directory(self, cache_path: Path):
        """Compress cache directory to save space."""
        try:
            # Create compressed archive
            archive_path = cache_path.with_suffix(".tar.gz")
            shutil.make_archive(str(archive_path.with_suffix("")), "gztar", cache_path)

            # Remove original directory
            shutil.rmtree(cache_path)

            # Rename archive
            archive_path.rename(cache_path)

        except Exception as e:
            logger.warning(f"Failed to compress cache directory: {e}")

    def _decompress_cache_directory(self, cache_path: Path):
        """Decompress cache directory if needed."""
        try:
            if cache_path.is_file():
                # Extract compressed archive
                import tarfile

                with tarfile.open(cache_path, "r:gz") as tar:
                    tar.extractall(cache_path.parent)

                # Remove compressed file
                cache_path.unlink()

        except Exception as e:
            logger.warning(f"Failed to decompress cache directory: {e}")

    def _load_model_from_disk(self, model_type: str, cache_path: Path, device: str, **kwargs) -> Optional[Any]:
        """Load a model from disk cache."""
        try:
            # Decompress if needed
            if self.compression_enabled:
                self._decompress_cache_directory(cache_path)

            if model_type == "clip":
                from transformers import CLIPProcessor, CLIPModel

                model = CLIPModel.from_pretrained(cache_path / "model")
                processor = CLIPProcessor.from_pretrained(cache_path / "processor")
                model.processor = processor
                return model.to(device)

            elif model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer

                return SentenceTransformer(str(cache_path), device=device)

            elif model_type == "stable_diffusion":
                from diffusers import StableDiffusionPipeline

                torch_dtype = kwargs.get("torch_dtype", torch.float16 if device == "cuda" else torch.float32)
                pipeline = StableDiffusionPipeline.from_pretrained(str(cache_path), torch_dtype=torch_dtype)
                return pipeline.to(device)

            else:
                return None

        except Exception as e:
            logger.error(f"Failed to load model from disk: {e}")
            return None

    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry from both memory and disk."""
        with self._lock:
            # Remove from memory cache
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                self._memory_cache_size -= entry.size_bytes
                del self._memory_cache[cache_key]

            # Remove from disk cache
            cache_path = self.cache_dir / cache_key
            if cache_path.exists():
                try:
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                    else:
                        cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove cache file: {e}")

            # Remove from metadata
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()

    def clear_cache(self, model_type: Optional[str] = None):
        """Clear cache entries, optionally filtered by model type."""
        with self._lock:
            keys_to_remove = []

            for cache_key, metadata in self.metadata.items():
                if model_type is None or metadata.get("model_type") == model_type:
                    keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                self._remove_cache_entry(cache_key)

            logger.info(f"Cleared {len(keys_to_remove)} cache entries")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information and statistics."""
        with self._lock:
            # Calculate hit rate
            total_requests = self.stats["total_requests"]
            hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0

            # Get cached models info
            cached_models = {}
            for cache_key, metadata in self.metadata.items():
                model_type = metadata.get("model_type", "unknown")
                model_name = metadata.get("model_name", "unknown")
                if model_type not in cached_models:
                    cached_models[model_type] = []
                cached_models[model_type].append(model_name)

            return {
                "memory_cache_size": len(self._memory_cache),
                "memory_cache_size_bytes": self._memory_cache_size,
                "memory_cache_size_mb": self._memory_cache_size / (1024 * 1024),
                "disk_cache_size": len(self.metadata),
                "disk_cache_size_bytes": self._get_disk_cache_size(),
                "disk_cache_size_mb": self._get_disk_cache_size() / (1024 * 1024),
                "total_size_bytes": self._memory_cache_size + self._get_disk_cache_size(),
                "total_size_mb": (self._memory_cache_size + self._get_disk_cache_size()) / (1024 * 1024),
                "cached_models": cached_models,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests,
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "evictions": self.stats["evictions"],
                "validation_failures": self.stats["validation_failures"],
                "compression_enabled": self.compression_enabled,
                "max_memory_size_mb": self.max_memory_size / (1024 * 1024),
                "max_disk_size_mb": self.max_disk_size / (1024 * 1024),
            }

    def warmup_cache(self, model_configs: List[Dict[str, Any]]):
        """Preload frequently used models into cache."""
        logger.info(f"Starting cache warmup for {len(model_configs)} models")

        for config in model_configs:
            try:
                model_type = config["model_type"]
                model_name = config["model_name"]
                device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

                # Try to load and cache the model
                cached_model = self.get_cached_model(model_type, model_name, device, **config.get("kwargs", {}))
                if cached_model is None:
                    logger.info(f"Model {model_name} not in cache, will be loaded on first use")
                else:
                    logger.info(f"Successfully warmed up {model_name}")

            except Exception as e:
                logger.warning(f"Failed to warmup {config.get('model_name', 'unknown')}: {e}")

    def reset_stats(self):
        """Reset cache statistics."""
        with self._lock:
            self.stats = self._get_default_stats()
            self._save_stats()
            logger.info("Cache statistics reset")

    def optimize_cache(self):
        """Optimize cache by removing rarely used entries and defragmenting."""
        with self._lock:
            logger.info("Starting cache optimization...")

            # Remove entries that haven't been accessed in a long time (30 days)
            cutoff_time = time.time() - (30 * 24 * 60 * 60)
            keys_to_remove = []

            for cache_key, metadata in self.metadata.items():
                last_accessed = metadata.get("last_accessed", 0)
                if last_accessed < cutoff_time:
                    keys_to_remove.append(cache_key)

            for cache_key in keys_to_remove:
                self._remove_cache_entry(cache_key)

            logger.info(f"Optimization complete: removed {len(keys_to_remove)} old entries")


# Global cache instance
_global_cache = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ModelCache()
    return _global_cache
