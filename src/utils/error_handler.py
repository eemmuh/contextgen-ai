"""
Comprehensive error handling system for the RAG-based Image Generation System.
"""

import time
import functools
from typing import Callable, Any, Optional, Dict
from enum import Enum
import threading
from dataclasses import dataclass
from src.utils.logger import get_logger, log_error_with_context

logger = get_logger("error_handler")


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    operation: str
    component: str
    severity: ErrorSeverity
    retryable: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryHandler:
    """Retry logic with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")

        raise last_exception


class ErrorHandler:
    """Main error handling system."""

    def __init__(self):
        """Initialize error handler."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.error_stats: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def handle_operation(self, context: ErrorContext):
        """Decorator for handling operations with error context."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                return self._execute_with_error_handling(func, context, *args, **kwargs)

            return wrapper

        return decorator

    def _execute_with_error_handling(self, func: Callable, context: ErrorContext, *args, **kwargs) -> Any:
        """Execute function with comprehensive error handling."""
        start_time = time.time()

        try:
            # Get or create circuit breaker
            circuit_breaker_key = f"{context.component}_{context.operation}"
            if circuit_breaker_key not in self.circuit_breakers:
                with self._lock:
                    if circuit_breaker_key not in self.circuit_breakers:
                        self.circuit_breakers[circuit_breaker_key] = CircuitBreaker(
                            failure_threshold=context.circuit_breaker_threshold,
                            timeout=context.circuit_breaker_timeout,
                        )

            circuit_breaker = self.circuit_breakers[circuit_breaker_key]

            # Execute with circuit breaker
            if context.retryable:
                # Get or create retry handler
                if circuit_breaker_key not in self.retry_handlers:
                    with self._lock:
                        if circuit_breaker_key not in self.retry_handlers:
                            self.retry_handlers[circuit_breaker_key] = RetryHandler(
                                max_retries=context.max_retries,
                                base_delay=context.retry_delay,
                            )

                retry_handler = self.retry_handlers[circuit_breaker_key]
                result = retry_handler.execute(circuit_breaker.call, func, *args, **kwargs)
            else:
                result = circuit_breaker.call(func, *args, **kwargs)

            # Log success
            duration = time.time() - start_time
            logger.info(f"Operation {context.operation} completed successfully in {duration:.3f}s")

            return result

        except Exception as e:
            # Log error with context
            duration = time.time() - start_time
            log_error_with_context(
                error=e,
                context=f"{context.component}.{context.operation}",
                duration=duration,
                severity=context.severity.value,
            )

            # Update error statistics
            self._update_error_stats(context, e)

            # Handle based on severity
            self._handle_error_by_severity(context, e)

            raise

    def _update_error_stats(self, context: ErrorContext, error: Exception):
        """Update error statistics."""
        with self._lock:
            key = f"{context.component}_{context.operation}"
            if key not in self.error_stats:
                self.error_stats[key] = {
                    "total_errors": 0,
                    "error_types": {},
                    "last_error_time": None,
                    "severity_counts": {},
                }

            stats = self.error_stats[key]
            stats["total_errors"] += 1
            stats["last_error_time"] = time.time()

            error_type = type(error).__name__
            stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1

            severity = context.severity.value
            stats["severity_counts"][severity] = stats["severity_counts"].get(severity, 0) + 1

    def _handle_error_by_severity(self, context: ErrorContext, error: Exception):
        """Handle error based on severity level."""
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR in {context.component}.{context.operation}: {str(error)}")
            # Could trigger alerts, shutdown procedures, etc.

        elif context.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH severity error in {context.component}.{context.operation}: {str(error)}")
            # Could trigger monitoring alerts

        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error in {context.component}.{context.operation}: {str(error)}")

        else:  # LOW
            logger.info(f"Low severity error in {context.component}.{context.operation}: {str(error)}")

    def get_error_stats(self) -> Dict[str, Dict]:
        """Get error statistics."""
        with self._lock:
            return self.error_stats.copy()

    def reset_error_stats(self):
        """Reset error statistics."""
        with self._lock:
            self.error_stats.clear()

    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get status of all circuit breakers."""
        return {key: cb.state for key, cb in self.circuit_breakers.items()}


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_operation(context: ErrorContext):
    """Decorator for handling operations with error context."""
    return get_error_handler().handle_operation(context)


def safe_execute(func: Callable, *args, default_return: Any = None, **kwargs) -> Any:
    """Safely execute a function with a default return value on error."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe execution failed for {func.__name__}: {str(e)}")
        return default_return


def with_fallback(fallback_func: Callable):
    """Decorator to provide fallback functionality."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function {func.__name__} failed, using fallback: {str(e)}")
                return fallback_func(*args, **kwargs)

        return wrapper

    return decorator
