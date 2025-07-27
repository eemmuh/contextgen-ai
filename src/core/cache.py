"""
Caching system for the application.

This module provides a comprehensive caching system with Redis support and
fallback to in-memory cache for improved performance and reduced API calls.
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging
import time

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from config.settings import get_settings
from src.core.exceptions import CacheError

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseCache:
    """Base cache interface."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        raise NotImplementedError


class MemoryCache(BaseCache):
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if entry.get("expires_at") is None:
            return False
        return datetime.utcnow() > entry["expires_at"]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove oldest entries (simple LRU-like behavior)
            oldest_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("created_at", datetime.min)
            )[:len(self._cache) - self.max_size + 1]
            
            for key in oldest_keys:
                del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup_expired()
        
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                self._hits += 1
                return entry["value"]
            else:
                del self._cache[key]
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            ttl = ttl or self.default_ttl
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            self._cache[key] = {
                "value": value,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at,
                "ttl": ttl
            }
            
            self._sets += 1
            self._evict_if_needed()
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if key in self._cache:
                del self._cache[key]
                self._deletes += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        self._cleanup_expired()
        return key in self._cache
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self._cache.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "type": "memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "deletes": self._deletes,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class RedisCache(BaseCache):
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if not REDIS_AVAILABLE:
            raise CacheError("Redis is not available. Install redis package.")
        
        try:
            self.redis_client = redis.from_url(redis_url)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            raise CacheError(f"Failed to connect to Redis: {e}")
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for Redis storage."""
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            # Fallback to pickle for complex objects
            return pickle.dumps(value).hex()
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from Redis storage."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Try pickle deserialization
            try:
                return pickle.loads(bytes.fromhex(value))
            except Exception:
                return value
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(key)
            if value is not None:
                return self._deserialize(value.decode())
            return None
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            serialized_value = self._serialize(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                return self.redis_client.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self.redis_client.flushdb()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"type": "redis", "error": str(e)}


class CacheManager:
    """Main cache manager that handles multiple cache backends."""
    
    def __init__(self):
        self._caches: Dict[str, BaseCache] = {}
        self._default_cache = "memory"
        self._initialize_caches()
    
    def _initialize_caches(self) -> None:
        """Initialize available cache backends."""
        # Always initialize memory cache as fallback
        self._caches["memory"] = MemoryCache(
            max_size=settings.cache.max_size,
            default_ttl=settings.cache.ttl
        )
        
        # Try to initialize Redis cache if available
        if settings.cache.enabled and REDIS_AVAILABLE:
            try:
                redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
                self._caches["redis"] = RedisCache(redis_url)
                self._default_cache = "redis"
                logger.info("Using Redis as primary cache")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}. Using memory cache.")
        
        logger.info(f"Cache manager initialized with {len(self._caches)} backends")
    
    def get(self, key: str, cache_name: Optional[str] = None) -> Optional[Any]:
        """Get value from cache."""
        cache_name = cache_name or self._default_cache
        cache = self._caches.get(cache_name)
        
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found, using default")
            cache = self._caches[self._default_cache]
        
        return cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_name: Optional[str] = None) -> bool:
        """Set value in cache."""
        cache_name = cache_name or self._default_cache
        cache = self._caches.get(cache_name)
        
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found, using default")
            cache = self._caches[self._default_cache]
        
        return cache.set(key, value, ttl)
    
    def delete(self, key: str, cache_name: Optional[str] = None) -> bool:
        """Delete value from cache."""
        cache_name = cache_name or self._default_cache
        cache = self._caches.get(cache_name)
        
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found, using default")
            cache = self._caches[self._default_cache]
        
        return cache.delete(key)
    
    def exists(self, key: str, cache_name: Optional[str] = None) -> bool:
        """Check if key exists in cache."""
        cache_name = cache_name or self._default_cache
        cache = self._caches.get(cache_name)
        
        if cache is None:
            logger.warning(f"Cache '{cache_name}' not found, using default")
            cache = self._caches[self._default_cache]
        
        return cache.exists(key)
    
    def clear(self, cache_name: Optional[str] = None) -> bool:
        """Clear cache."""
        if cache_name:
            cache = self._caches.get(cache_name)
            if cache is None:
                logger.warning(f"Cache '{cache_name}' not found")
                return False
            return cache.clear()
        else:
            # Clear all caches
            success = True
            for cache in self._caches.values():
                if not cache.clear():
                    success = False
            return success
    
    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if cache_name:
            cache = self._caches.get(cache_name)
            if cache is None:
                return {"error": f"Cache '{cache_name}' not found"}
            return cache.get_stats()
        else:
            # Get stats for all caches
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }
    
    def get_available_caches(self) -> List[str]:
        """Get list of available cache backends."""
        return list(self._caches.keys())


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_result(ttl: Optional[int] = None, key_prefix: str = "", cache_name: Optional[str] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add args and kwargs to key
            if args:
                key_parts.append(str(hash(str(args))))
            if kwargs:
                key_parts.append(str(hash(str(sorted(kwargs.items())))))
            
            cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = cache_manager.get(cache_key, cache_name)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl, cache_name)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str, cache_name: Optional[str] = None):
    """Decorator to invalidate cache entries after function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache entries matching pattern
            cache_manager = get_cache_manager()
            # Note: This is a simplified implementation
            # In a real scenario, you might want to implement pattern-based deletion
            
            return result
        
        return wrapper
    return decorator 