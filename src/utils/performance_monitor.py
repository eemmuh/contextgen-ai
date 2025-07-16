"""
Performance monitoring and profiling system for the RAG-based Image Generation System.
"""

import time
import psutil
import threading
import functools
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path
import gc
import torch
from src.utils.logger import get_logger, log_performance, log_memory_usage

logger = get_logger('performance')

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    operation: str
    component: str
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, history_size: int = 1000, enable_profiling: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of metrics to keep in history
            enable_profiling: Whether to enable detailed profiling
        """
        self.history_size = history_size
        self.enable_profiling = enable_profiling
        
        # Performance metrics storage
        self.metrics: deque = deque(maxlen=history_size)
        self.component_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # Memory tracking
        self.memory_snapshots: List[Dict] = []
        self.memory_threshold_mb = 1024  # 1GB threshold
        
        # CPU tracking
        self.cpu_usage: List[float] = []
        self.cpu_threshold = 80.0  # 80% threshold
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start background monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._background_monitor,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Background monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Background monitoring stopped")
    
    def _background_monitor(self, interval: float):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._capture_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
    
    def _capture_system_metrics(self):
        """Capture current system metrics."""
        process = psutil.Process()
        
        # Memory metrics
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # CPU metrics
        cpu_percent = process.cpu_percent()
        
        # GPU metrics (if available)
        gpu_metrics = self._get_gpu_metrics()
        
        snapshot = {
            'timestamp': time.time(),
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'gpu_metrics': gpu_metrics
        }
        
        with self._lock:
            self.memory_snapshots.append(snapshot)
            self.cpu_usage.append(cpu_percent)
            
            # Keep only recent snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots.pop(0)
            if len(self.cpu_usage) > 100:
                self.cpu_usage.pop(0)
        
        # Check thresholds
        if memory_mb > self.memory_threshold_mb:
            logger.warning(f"High memory usage: {memory_mb:.1f} MB")
        
        if cpu_percent > self.cpu_threshold:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU metrics if available."""
        try:
            if torch.cuda.is_available():
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    gpu_metrics[f'gpu_{i}'] = {
                        'memory_allocated_mb': torch.cuda.memory_allocated(i) / 1024 / 1024,
                        'memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024 / 1024,
                        'memory_free_mb': torch.cuda.memory_reserved(i) / 1024 / 1024 - 
                                        torch.cuda.memory_allocated(i) / 1024 / 1024
                    }
                return gpu_metrics
        except Exception as e:
            logger.debug(f"Could not get GPU metrics: {e}")
        
        return {}
    
    def monitor_operation(self, operation: str, component: str = "general"):
        """Decorator to monitor operation performance."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                return self._monitor_function_call(func, operation, component, *args, **kwargs)
            return wrapper
        return decorator
    
    def _monitor_function_call(self, func: Callable, operation: str, component: str, *args, **kwargs) -> Any:
        """Monitor a function call and record metrics."""
        start_time = time.time()
        
        # Capture initial state
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        cpu_before = process.cpu_percent()
        
        # Track peak memory
        peak_memory = memory_before
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Capture final state
            memory_after = process.memory_info().rss / 1024 / 1024
            cpu_after = process.cpu_percent()
            duration = time.time() - start_time
            
            # Update peak memory
            peak_memory = max(peak_memory, memory_after)
            
            # Create metric
            metric = PerformanceMetric(
                operation=operation,
                component=component,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=peak_memory,
                cpu_percent=(cpu_before + cpu_after) / 2,
                timestamp=start_time,
                metadata={
                    'function_name': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
            )
            
            # Store metric
            self._store_metric(metric)
            
            # Log performance
            log_performance(operation, duration, component=component, memory_delta=memory_after - memory_before)
            
            return result
            
        except Exception as e:
            # Capture error state
            memory_after = process.memory_info().rss / 1024 / 1024
            duration = time.time() - start_time
            
            # Create error metric
            metric = PerformanceMetric(
                operation=operation,
                component=component,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=peak_memory,
                cpu_percent=cpu_before,
                timestamp=start_time,
                metadata={
                    'function_name': func.__name__,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            )
            
            self._store_metric(metric)
            raise
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        with self._lock:
            self.metrics.append(metric)
            self.component_metrics[metric.component].append(metric)
            
            # Keep component metrics within limit
            if len(self.component_metrics[metric.component]) > self.history_size:
                self.component_metrics[metric.component].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self.metrics:
                return {}
            
            # Calculate overall statistics
            durations = [m.duration for m in self.metrics]
            memory_deltas = [m.memory_after - m.memory_before for m in self.metrics]
            cpu_usage = [m.cpu_percent for m in self.metrics]
            
            summary = {
                'total_operations': len(self.metrics),
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta': max(memory_deltas),
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_usage': max(cpu_usage),
                'components': {}
            }
            
            # Component-specific statistics
            for component, metrics in self.component_metrics.items():
                if metrics:
                    comp_durations = [m.duration for m in metrics]
                    comp_memory_deltas = [m.memory_after - m.memory_before for m in metrics]
                    
                    summary['components'][component] = {
                        'operation_count': len(metrics),
                        'avg_duration': sum(comp_durations) / len(comp_durations),
                        'max_duration': max(comp_durations),
                        'avg_memory_delta': sum(comp_memory_deltas) / len(comp_memory_deltas),
                        'max_memory_delta': max(comp_memory_deltas)
                    }
            
            return summary
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        process = psutil.Process()
        
        return {
            'timestamp': time.time(),
            'memory': {
                'rss_mb': process.memory_info().rss / 1024 / 1024,
                'vms_mb': process.memory_info().vms / 1024 / 1024,
                'percent': process.memory_percent()
            },
            'cpu': {
                'percent': process.cpu_percent(),
                'count': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'gpu': self._get_gpu_metrics(),
            'disk': {
                'usage_percent': psutil.disk_usage('/').percent
            }
        }
    
    def optimize_memory(self):
        """Perform memory optimization."""
        logger.info("Starting memory optimization...")
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared PyTorch CUDA cache")
        
        # Get memory after optimization
        process = psutil.Process()
        memory_after = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after optimization: {memory_after:.1f} MB")
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with self._lock:
            data = {
                'summary': self.get_performance_summary(),
                'system_metrics': self.get_system_metrics(),
                'recent_metrics': [
                    {
                        'operation': m.operation,
                        'component': m.component,
                        'duration': m.duration,
                        'memory_before': m.memory_before,
                        'memory_after': m.memory_after,
                        'memory_peak': m.memory_peak,
                        'cpu_percent': m.cpu_percent,
                        'timestamp': m.timestamp,
                        'metadata': m.metadata
                    }
                    for m in list(self.metrics)[-100:]  # Last 100 metrics
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics exported to {filepath}")

# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def monitor_operation(operation: str, component: str = "general"):
    """Decorator to monitor operation performance."""
    return get_performance_monitor().monitor_operation(operation, component)

def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary."""
    return get_performance_monitor().get_performance_summary()

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    return get_performance_monitor().get_system_metrics()

def optimize_memory():
    """Optimize memory usage."""
    get_performance_monitor().optimize_memory()

def export_metrics(filepath: str):
    """Export performance metrics."""
    get_performance_monitor().export_metrics(filepath) 