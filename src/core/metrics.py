"""
Metrics collection and monitoring system.

This module provides comprehensive metrics collection for monitoring
application performance, usage patterns, and system health.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricValue:
    """Metric value with timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str]


@dataclass
class Metric:
    """Metric definition."""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = None


class MetricsCollector:
    """Main metrics collector."""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.metrics: Dict[str, Metric] = {}
        self.values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.lock = threading.RLock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        self.register_metric(
            "api_requests_total",
            MetricType.COUNTER,
            "Total number of API requests",
            "requests"
        )
        
        self.register_metric(
            "api_request_duration_seconds",
            MetricType.HISTOGRAM,
            "API request duration",
            "seconds"
        )
        
        self.register_metric(
            "api_requests_active",
            MetricType.GAUGE,
            "Number of active API requests",
            "requests"
        )
        
        self.register_metric(
            "embedding_generation_total",
            MetricType.COUNTER,
            "Total number of embedding generations",
            "embeddings"
        )
        
        self.register_metric(
            "embedding_generation_duration_seconds",
            MetricType.HISTOGRAM,
            "Embedding generation duration",
            "seconds"
        )
        
        self.register_metric(
            "image_generation_total",
            MetricType.COUNTER,
            "Total number of image generations",
            "images"
        )
        
        self.register_metric(
            "image_generation_duration_seconds",
            MetricType.HISTOGRAM,
            "Image generation duration",
            "seconds"
        )
        
        self.register_metric(
            "cache_hits_total",
            MetricType.COUNTER,
            "Total number of cache hits",
            "hits"
        )
        
        self.register_metric(
            "cache_misses_total",
            MetricType.COUNTER,
            "Total number of cache misses",
            "misses"
        )
        
        self.register_metric(
            "database_connections_active",
            MetricType.GAUGE,
            "Number of active database connections",
            "connections"
        )
        
        self.register_metric(
            "database_query_duration_seconds",
            MetricType.HISTOGRAM,
            "Database query duration",
            "seconds"
        )
        
        self.register_metric(
            "memory_usage_bytes",
            MetricType.GAUGE,
            "Memory usage in bytes",
            "bytes"
        )
        
        self.register_metric(
            "cpu_usage_percent",
            MetricType.GAUGE,
            "CPU usage percentage",
            "percent"
        )
    
    def register_metric(self, name: str, metric_type: MetricType, description: str, unit: str = "", labels: Dict[str, str] = None):
        """Register a new metric."""
        with self.lock:
            if name in self.metrics:
                logger.warning(f"Metric {name} already registered, overwriting")
            
            self.metrics[name] = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                labels=labels or {}
            )
    
    def record_value(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value."""
        with self.lock:
            if name not in self.metrics:
                logger.warning(f"Recording value for unregistered metric: {name}")
                self.register_metric(name, MetricType.GAUGE, f"Auto-registered metric: {name}")
            
            metric_value = MetricValue(
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            
            self.values[name].append(metric_value)
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_value(name, value, labels)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        self.record_value(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric value."""
        self.record_value(name, value, labels)
    
    def time_operation(self, name: str, labels: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)
    
    def get_metric_stats(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistics for a metric over a time window."""
        with self.lock:
            if name not in self.metrics:
                return {"error": f"Metric {name} not found"}
            
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            values = [
                mv.value for mv in self.values[name]
                if mv.timestamp >= cutoff_time
            ]
            
            if not values:
                return {
                    "metric": name,
                    "window_minutes": window_minutes,
                    "count": 0,
                    "min": None,
                    "max": None,
                    "avg": None,
                    "sum": 0
                }
            
            return {
                "metric": name,
                "window_minutes": window_minutes,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "sum": sum(values)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and their current values."""
        with self.lock:
            result = {}
            for name, metric in self.metrics.items():
                values = list(self.values[name])
                if values:
                    latest_value = values[-1]
                    result[name] = {
                        "metric": asdict(metric),
                        "latest_value": {
                            "value": latest_value.value,
                            "timestamp": latest_value.timestamp.isoformat(),
                            "labels": latest_value.labels
                        },
                        "total_values": len(values)
                    }
                else:
                    result[name] = {
                        "metric": asdict(metric),
                        "latest_value": None,
                        "total_values": 0
                    }
            return result
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        with self.lock:
            lines = []
            
            for name, metric in self.metrics.items():
                values = list(self.values[name])
                if not values:
                    continue
                
                # Get the latest value for each metric
                latest_value = values[-1]
                
                # Build labels string
                labels_str = ""
                if latest_value.labels:
                    label_parts = [f'{k}="{v}"' for k, v in latest_value.labels.items()]
                    labels_str = "{" + ",".join(label_parts) + "}"
                
                # Add help and type comments
                lines.append(f"# HELP {name} {metric.description}")
                lines.append(f"# TYPE {name} {metric.type.value}")
                
                # Add metric value
                lines.append(f"{name}{labels_str} {latest_value.value}")
            
            return "\n".join(lines)
    
    def clear_old_data(self, max_age_hours: int = 24):
        """Clear old metric data."""
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            
            for name in list(self.values.keys()):
                # Remove old values
                self.values[name] = deque(
                    (mv for mv in self.values[name] if mv.timestamp >= cutoff_time),
                    maxlen=self.max_history_size
                )


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.record_histogram(self.name, duration, self.labels)


class MetricsMiddleware:
    """FastAPI middleware for collecting request metrics."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
    
    async def __call__(self, request, call_next):
        start_time = time.time()
        
        # Increment active requests
        self.collector.increment_counter("api_requests_active", 1)
        
        try:
            response = await call_next(request)
            
            # Record request metrics
            duration = time.time() - start_time
            self.collector.increment_counter("api_requests_total", 1, {
                "method": request.method,
                "path": request.url.path,
                "status_code": str(response.status_code)
            })
            self.collector.record_histogram("api_request_duration_seconds", duration, {
                "method": request.method,
                "path": request.url.path
            })
            
            return response
        finally:
            # Decrement active requests
            self.collector.increment_counter("api_requests_active", -1)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_metric(name: str, value: float, labels: Dict[str, str] = None):
    """Record a metric value."""
    collector = get_metrics_collector()
    collector.record_value(name, value, labels)


def increment_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None):
    """Increment a counter metric."""
    collector = get_metrics_collector()
    collector.increment_counter(name, value, labels)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Set a gauge metric value."""
    collector = get_metrics_collector()
    collector.set_gauge(name, value, labels)


def time_operation(name: str, labels: Dict[str, str] = None):
    """Context manager for timing operations."""
    collector = get_metrics_collector()
    return collector.time_operation(name, labels)


def metrics_decorator(metric_name: str, metric_type: MetricType = MetricType.COUNTER, labels: Dict[str, str] = None):
    """Decorator for automatically recording metrics."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            
            if metric_type == MetricType.TIMER:
                with collector.time_operation(metric_name, labels):
                    return func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                collector.record_value(metric_name, 1.0, labels)
                return result
        
        return wrapper
    return decorator 