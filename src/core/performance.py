"""
Performance optimization and resource management.

This module provides performance optimizations including connection pooling,
async operations, resource management, and performance monitoring.
"""

import asyncio
import threading
import time
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager, contextmanager
import logging
from dataclasses import dataclass
from enum import Enum
import weakref

from src.core.metrics import get_metrics_collector, time_operation
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ResourceType(Enum):
    """Types of resources that can be pooled."""
    DATABASE = "database"
    HTTP = "http"
    MODEL = "model"
    CACHE = "cache"


@dataclass
class PoolConfig:
    """Configuration for connection pools."""
    max_size: int = 10
    min_size: int = 2
    max_overflow: int = 20
    timeout: float = 30.0
    recycle: float = 3600.0  # Recycle connections after 1 hour
    pre_ping: bool = True


class ConnectionPool(Generic[T]):
    """Generic connection pool implementation."""
    
    def __init__(self, resource_type: ResourceType, config: PoolConfig, factory: Callable[[], T]):
        self.resource_type = resource_type
        self.config = config
        self.factory = factory
        self._pool: List[T] = []
        self._in_use: List[T] = []
        self._lock = threading.RLock()
        self._created_at = time.time()
        self._last_cleanup = time.time()
        
        # Initialize pool with minimum connections
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the pool with minimum connections."""
        for _ in range(self.config.min_size):
            try:
                connection = self.factory()
                self._pool.append(connection)
            except Exception as e:
                logger.error(f"Failed to create initial connection for {self.resource_type.value}: {e}")
    
    def _cleanup_expired_connections(self):
        """Clean up expired connections."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return
        
        self._last_cleanup = current_time
        
        with self._lock:
            # Remove connections older than recycle time
            expired_connections = [
                conn for conn in self._pool
                if hasattr(conn, '_created_at') and 
                current_time - conn._created_at > self.config.recycle
            ]
            
            for conn in expired_connections:
                self._pool.remove(conn)
                self._close_connection(conn)
    
    def _close_connection(self, connection: T):
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                connection.close()
            elif hasattr(connection, 'disconnect'):
                connection.disconnect()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    def get_connection(self) -> T:
        """Get a connection from the pool."""
        self._cleanup_expired_connections()
        
        with self._lock:
            # Try to get from pool
            if self._pool:
                connection = self._pool.pop()
                self._in_use.append(connection)
                return connection
            
            # Create new connection if under max size
            if len(self._in_use) < self.config.max_size + self.config.max_overflow:
                try:
                    connection = self.factory()
                    connection._created_at = time.time()
                    self._in_use.append(connection)
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create new connection: {e}")
                    raise
        
        # Wait for available connection
        start_time = time.time()
        while time.time() - start_time < self.config.timeout:
            time.sleep(0.1)
            with self._lock:
                if self._pool:
                    connection = self._pool.pop()
                    self._in_use.append(connection)
                    return connection
        
        raise TimeoutError(f"Timeout waiting for {self.resource_type.value} connection")
    
    def return_connection(self, connection: T):
        """Return a connection to the pool."""
        with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)
                
                # Check if connection is still valid
                if self.config.pre_ping and hasattr(connection, 'ping'):
                    try:
                        connection.ping()
                    except Exception:
                        self._close_connection(connection)
                        return
                
                # Return to pool if under max size
                if len(self._pool) < self.config.max_size:
                    self._pool.append(connection)
                else:
                    self._close_connection(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "resource_type": self.resource_type.value,
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "total_connections": len(self._pool) + len(self._in_use),
                "max_size": self.config.max_size,
                "max_overflow": self.config.max_overflow,
                "uptime_seconds": time.time() - self._created_at
            }
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._pool + self._in_use:
                self._close_connection(conn)
            self._pool.clear()
            self._in_use.clear()


class AsyncConnectionPool(Generic[T]):
    """Async connection pool implementation."""
    
    def __init__(self, resource_type: ResourceType, config: PoolConfig, factory: Callable[[], T]):
        self.resource_type = resource_type
        self.config = config
        self.factory = factory
        self._pool: List[T] = []
        self._in_use: List[T] = []
        self._lock = asyncio.Lock()
        self._created_at = time.time()
        self._last_cleanup = time.time()
        
        # Initialize pool
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self):
        """Initialize the pool with minimum connections."""
        for _ in range(self.config.min_size):
            try:
                connection = await self.factory()
                self._pool.append(connection)
            except Exception as e:
                logger.error(f"Failed to create initial async connection for {self.resource_type.value}: {e}")
    
    async def _cleanup_expired_connections(self):
        """Clean up expired connections."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return
        
        self._last_cleanup = current_time
        
        async with self._lock:
            # Remove connections older than recycle time
            expired_connections = [
                conn for conn in self._pool
                if hasattr(conn, '_created_at') and 
                current_time - conn._created_at > self.config.recycle
            ]
            
            for conn in expired_connections:
                self._pool.remove(conn)
                await self._close_connection(conn)
    
    async def _close_connection(self, connection: T):
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            elif hasattr(connection, 'disconnect'):
                await connection.disconnect()
            elif hasattr(connection, 'close'):
                connection.close()
        except Exception as e:
            logger.warning(f"Error closing async connection: {e}")
    
    async def get_connection(self) -> T:
        """Get a connection from the pool."""
        await self._cleanup_expired_connections()
        
        async with self._lock:
            # Try to get from pool
            if self._pool:
                connection = self._pool.pop()
                self._in_use.append(connection)
                return connection
            
            # Create new connection if under max size
            if len(self._in_use) < self.config.max_size + self.config.max_overflow:
                try:
                    connection = await self.factory()
                    connection._created_at = time.time()
                    self._in_use.append(connection)
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create new async connection: {e}")
                    raise
        
        # Wait for available connection
        start_time = time.time()
        while time.time() - start_time < self.config.timeout:
            await asyncio.sleep(0.1)
            async with self._lock:
                if self._pool:
                    connection = self._pool.pop()
                    self._in_use.append(connection)
                    return connection
        
        raise TimeoutError(f"Timeout waiting for {self.resource_type.value} connection")
    
    async def return_connection(self, connection: T):
        """Return a connection to the pool."""
        async with self._lock:
            if connection in self._in_use:
                self._in_use.remove(connection)
                
                # Check if connection is still valid
                if self.config.pre_ping and hasattr(connection, 'ping'):
                    try:
                        await connection.ping()
                    except Exception:
                        await self._close_connection(connection)
                        return
                
                # Return to pool if under max size
                if len(self._pool) < self.config.max_size:
                    self._pool.append(connection)
                else:
                    await self._close_connection(connection)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        async with self._lock:
            return {
                "resource_type": self.resource_type.value,
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "total_connections": len(self._pool) + len(self._in_use),
                "max_size": self.config.max_size,
                "max_overflow": self.config.max_overflow,
                "uptime_seconds": time.time() - self._created_at
            }
    
    async def close_all(self):
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool + self._in_use:
                await self._close_connection(conn)
            self._pool.clear()
            self._in_use.clear()


class PerformanceOptimizer:
    """Main performance optimization manager."""
    
    def __init__(self):
        self.pools: Dict[ResourceType, ConnectionPool] = {}
        self.async_pools: Dict[ResourceType, AsyncConnectionPool] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.metrics = get_metrics_collector()
    
    def register_pool(self, resource_type: ResourceType, config: PoolConfig, factory: Callable[[], T]) -> ConnectionPool[T]:
        """Register a new connection pool."""
        pool = ConnectionPool(resource_type, config, factory)
        self.pools[resource_type] = pool
        logger.info(f"Registered connection pool for {resource_type.value}")
        return pool
    
    def register_async_pool(self, resource_type: ResourceType, config: PoolConfig, factory: Callable[[], T]) -> AsyncConnectionPool[T]:
        """Register a new async connection pool."""
        pool = AsyncConnectionPool(resource_type, config, factory)
        self.async_pools[resource_type] = pool
        logger.info(f"Registered async connection pool for {resource_type.value}")
        return pool
    
    def get_pool(self, resource_type: ResourceType) -> Optional[ConnectionPool]:
        """Get a connection pool."""
        return self.pools.get(resource_type)
    
    def get_async_pool(self, resource_type: ResourceType) -> Optional[AsyncConnectionPool]:
        """Get an async connection pool."""
        return self.async_pools.get(resource_type)
    
    def run_in_thread_pool(self, func: Callable, *args, **kwargs):
        """Run a function in the thread pool."""
        with time_operation("thread_pool_execution_time"):
            return self.thread_pool.submit(func, *args, **kwargs)
    
    def run_in_process_pool(self, func: Callable, *args, **kwargs):
        """Run a function in the process pool."""
        with time_operation("process_pool_execution_time"):
            return self.process_pool.submit(func, *args, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        stats = {
            "pools": {},
            "async_pools": {},
            "thread_pool": {
                "max_workers": self.thread_pool._max_workers,
                "active_threads": len(self.thread_pool._threads),
                "queue_size": self.thread_pool._work_queue.qsize()
            },
            "process_pool": {
                "max_workers": self.process_pool._max_workers,
                "active_processes": len(self.process_pool._processes)
            }
        }
        
        for resource_type, pool in self.pools.items():
            stats["pools"][resource_type.value] = pool.get_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up all resources."""
        for pool in self.pools.values():
            pool.close_all()
        
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("Performance optimizer cleaned up")


# Global performance optimizer instance
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


@contextmanager
def connection_from_pool(resource_type: ResourceType):
    """Context manager for getting connections from pool."""
    optimizer = get_performance_optimizer()
    pool = optimizer.get_pool(resource_type)
    
    if pool is None:
        raise ValueError(f"No pool registered for {resource_type.value}")
    
    connection = pool.get_connection()
    try:
        yield connection
    finally:
        pool.return_connection(connection)


@asynccontextmanager
async def async_connection_from_pool(resource_type: ResourceType):
    """Async context manager for getting connections from pool."""
    optimizer = get_performance_optimizer()
    pool = optimizer.get_async_pool(resource_type)
    
    if pool is None:
        raise ValueError(f"No async pool registered for {resource_type.value}")
    
    connection = await pool.get_connection()
    try:
        yield connection
    finally:
        await pool.return_connection(connection)


def optimize_performance(func: Callable) -> Callable:
    """Decorator for performance optimization."""
    def wrapper(*args, **kwargs):
        with time_operation(f"{func.__name__}_execution_time"):
            return func(*args, **kwargs)
    return wrapper


async def optimize_async_performance(func: Callable) -> Callable:
    """Decorator for async performance optimization."""
    async def wrapper(*args, **kwargs):
        with time_operation(f"{func.__name__}_async_execution_time"):
            return await func(*args, **kwargs)
    return wrapper


class ResourceManager:
    """Resource management and cleanup."""
    
    def __init__(self):
        self._resources: List[weakref.ref] = []
        self._cleanup_handlers: List[Callable] = []
    
    def register_resource(self, resource: Any, cleanup_handler: Optional[Callable] = None):
        """Register a resource for cleanup."""
        self._resources.append(weakref.ref(resource))
        if cleanup_handler:
            self._cleanup_handlers.append(cleanup_handler)
    
    def cleanup(self):
        """Clean up all registered resources."""
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error during resource cleanup: {e}")
        
        self._resources.clear()
        self._cleanup_handlers.clear()


# Global resource manager instance
_resource_manager = ResourceManager()


def register_resource(resource: Any, cleanup_handler: Optional[Callable] = None):
    """Register a resource for cleanup."""
    _resource_manager.register_resource(resource, cleanup_handler)


def cleanup_resources():
    """Clean up all registered resources."""
    _resource_manager.cleanup() 