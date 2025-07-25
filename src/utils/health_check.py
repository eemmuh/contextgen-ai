"""
Health check and diagnostics system for the RAG-based Image Generation System.
"""

import time
import psutil
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json
import torch
from src.utils.logger import get_logger
from src.utils.model_cache import get_model_cache
from src.utils.performance_monitor import get_system_metrics

logger = get_logger("health_check")


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result data structure."""

    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    response_time: float


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.thresholds = {
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 90.0,
            "disk_usage_percent": 90.0,
            "gpu_memory_percent": 95.0,
            "cache_hit_rate": 50.0,
            "response_time_ms": 5000.0,
        }

        # Register default health checks
        self._register_default_checks()

        logger.info("Health checker initialized")

    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("gpu_status", self._check_gpu_status)
        self.register_check("model_cache", self._check_model_cache)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("cpu_usage", self._check_cpu_usage)
        self.register_check("torch_status", self._check_torch_status)

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def run_health_check(self, check_name: Optional[str] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks."""
        start_time = time.time()
        results = {}

        if check_name:
            # Run specific check
            if check_name in self.checks:
                result = self._run_single_check(check_name, self.checks[check_name])
                results[check_name] = result
            else:
                logger.warning(f"Health check '{check_name}' not found")
        else:
            # Run all checks
            for name, check_func in self.checks.items():
                result = self._run_single_check(name, check_func)
                results[name] = result

        # Store results
        self.results.update(results)

        total_time = time.time() - start_time
        logger.info(f"Health check completed in {total_time:.3f}s")

        return results

    def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()

        try:
            status, message, details = check_func()
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            result = HealthCheckResult(
                component=name,
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                response_time=response_time,
            )

            # Log result
            if status == HealthStatus.CRITICAL:
                logger.error(f"Health check {name}: {message}")
            elif status == HealthStatus.WARNING:
                logger.warning(f"Health check {name}: {message}")
            else:
                logger.info(f"Health check {name}: {message}")

            return result

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"Health check {name} failed: {e}")

            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                response_time=response_time,
            )

    def _check_system_resources(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check overall system resources."""
        try:
            system_metrics = get_system_metrics()

            memory_percent = system_metrics["memory"]["percent"]
            cpu_percent = system_metrics["cpu"]["percent"]
            disk_percent = system_metrics["disk"]["usage_percent"]

            # Determine overall status
            if (
                memory_percent > self.thresholds["memory_usage_percent"]
                or cpu_percent > self.thresholds["cpu_usage_percent"]
                or disk_percent > self.thresholds["disk_usage_percent"]
            ):
                status = HealthStatus.CRITICAL
                message = "System resources critically low"
            elif memory_percent > 70 or cpu_percent > 80 or disk_percent > 80:
                status = HealthStatus.WARNING
                message = "System resources getting low"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources healthy"

            details = {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk_percent,
                "thresholds": self.thresholds,
            }

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check system resources: {e}", {}

    def _check_gpu_status(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check GPU status and memory."""
        try:
            if not torch.cuda.is_available():
                return HealthStatus.HEALTHY, "No GPU available (CPU mode)", {}

            gpu_count = torch.cuda.device_count()
            gpu_details = {}

            for i in range(gpu_count):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024  # MB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024  # MB
                memory_usage_percent = (memory_reserved / memory_total) * 100

                gpu_details[f"gpu_{i}"] = {
                    "memory_allocated_mb": memory_allocated,
                    "memory_reserved_mb": memory_reserved,
                    "memory_total_mb": memory_total,
                    "memory_usage_percent": memory_usage_percent,
                }

            # Check if any GPU is over threshold
            max_usage = max(gpu["memory_usage_percent"] for gpu in gpu_details.values())

            if max_usage > self.thresholds["gpu_memory_percent"]:
                status = HealthStatus.CRITICAL
                message = f"GPU memory usage critical: {max_usage:.1f}%"
            elif max_usage > 80:
                status = HealthStatus.WARNING
                message = f"GPU memory usage high: {max_usage:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"GPU status healthy ({gpu_count} devices)"

            return status, message, {"gpu_count": gpu_count, "gpu_details": gpu_details}

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check GPU status: {e}", {}

    def _check_model_cache(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check model cache health."""
        try:
            cache = get_model_cache()
            cache_info = cache.get_cache_info()

            hit_rate = cache_info["hit_rate_percent"]
            total_requests = cache_info["total_requests"]
            memory_usage = cache_info["memory_cache_size_mb"]
            disk_usage = cache_info["disk_cache_size_mb"]

            if total_requests == 0:
                return HealthStatus.HEALTHY, "Cache not yet used", cache_info

            if hit_rate < self.thresholds["cache_hit_rate"]:
                status = HealthStatus.WARNING
                message = f"Cache hit rate low: {hit_rate:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Cache healthy (hit rate: {hit_rate:.1f}%)"

            details = {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "memory_usage_mb": memory_usage,
                "disk_usage_mb": disk_usage,
                "cache_hits": cache_info["cache_hits"],
                "cache_misses": cache_info["cache_misses"],
            }

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check model cache: {e}", {}

    def _check_disk_space(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage("/")
            usage_percent = disk_usage.percent
            free_gb = disk_usage.free / 1024 / 1024 / 1024

            if usage_percent > self.thresholds["disk_usage_percent"]:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {usage_percent:.1f}% used"
            elif usage_percent > 80:
                status = HealthStatus.WARNING
                message = f"Disk space getting low: {usage_percent:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {usage_percent:.1f}% used"

            details = {
                "usage_percent": usage_percent,
                "free_gb": free_gb,
                "total_gb": disk_usage.total / 1024 / 1024 / 1024,
                "used_gb": disk_usage.used / 1024 / 1024 / 1024,
            }

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check disk space: {e}", {}

    def _check_memory_usage(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            memory_mb = memory_info.rss / 1024 / 1024

            if memory_percent > self.thresholds["memory_usage_percent"]:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 70:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage healthy: {memory_percent:.1f}%"

            details = {
                "memory_percent": memory_percent,
                "memory_mb": memory_mb,
                "virtual_memory_mb": memory_info.vms / 1024 / 1024,
            }

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check memory usage: {e}", {}

    def _check_cpu_usage(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            if cpu_percent > self.thresholds["cpu_usage_percent"]:
                status = HealthStatus.CRITICAL
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage healthy: {cpu_percent:.1f}%"

            details = {"cpu_percent": cpu_percent, "cpu_count": cpu_count}

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check CPU usage: {e}", {}

    def _check_torch_status(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check PyTorch status."""
        try:
            details = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (torch.version.cuda if torch.cuda.is_available() else None),
                "device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
            }

            if torch.cuda.is_available():
                status = HealthStatus.HEALTHY
                message = f"PyTorch healthy with CUDA support ({details['device_count']} devices)"
            else:
                status = HealthStatus.WARNING
                message = "PyTorch running in CPU mode (no CUDA)"

            return status, message, details

        except Exception as e:
            return HealthStatus.UNKNOWN, f"Could not check PyTorch status: {e}", {}

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self.results.values()]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health check summary."""
        overall_status = self.get_overall_health()

        summary = {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "total_checks": len(self.results),
            "healthy_checks": len([r for r in self.results.values() if r.status == HealthStatus.HEALTHY]),
            "warning_checks": len([r for r in self.results.values() if r.status == HealthStatus.WARNING]),
            "critical_checks": len([r for r in self.results.values() if r.status == HealthStatus.CRITICAL]),
            "unknown_checks": len([r for r in self.results.values() if r.status == HealthStatus.UNKNOWN]),
            "results": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time_ms": result.response_time,
                    "timestamp": result.timestamp,
                }
                for name, result in self.results.items()
            },
        }

        return summary

    def export_health_report(self, filepath: str):
        """Export health check report to JSON."""
        report = {
            "summary": self.get_health_summary(),
            "detailed_results": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "response_time_ms": result.response_time,
                    "timestamp": result.timestamp,
                }
                for name, result in self.results.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Health report exported to {filepath}")


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def run_health_check(check_name: Optional[str] = None) -> Dict[str, HealthCheckResult]:
    """Run health checks."""
    return get_health_checker().run_health_check(check_name)


def get_health_summary() -> Dict[str, Any]:
    """Get health check summary."""
    return get_health_checker().get_health_summary()


def get_overall_health() -> HealthStatus:
    """Get overall system health status."""
    return get_health_checker().get_overall_health()


def export_health_report(filepath: str):
    """Export health check report."""
    get_health_checker().export_health_report(filepath)
