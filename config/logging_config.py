"""
Logging configuration for the application.

This module provides centralized logging configuration with support
for different log levels, formats, and outputs (console, file, etc.).
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import get_settings


def setup_logging(
    level: str = "INFO",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Setup application logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format string
        log_file: Path to log file (optional)
        max_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Enable console logging
        file_output: Enable file logging
    """
    # Get settings
    settings = get_settings()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    _configure_logger_levels()


def _configure_logger_levels() -> None:
    """Configure specific logger levels for different components."""
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Set application loggers to DEBUG in development
    settings = get_settings()
    if settings.is_development():
        logging.getLogger("src").setLevel(logging.DEBUG)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_application_logging() -> None:
    """Setup logging for the application using settings."""
    settings = get_settings()
    
    # Determine log file path
    log_file = None
    if settings.logging.file:
        log_file = str(settings.get_log_file_path())
    
    # Setup logging
    setup_logging(
        level=settings.logging.level,
        format_string=settings.logging.format,
        log_file=log_file,
        max_size=settings.logging.max_size,
        backup_count=settings.logging.backup_count,
        console_output=True,
        file_output=bool(settings.logging.file)
    )


# Structured logging for better parsing
class StructuredFormatter(logging.Formatter):
    """Structured log formatter for JSON-like output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured output."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return str(log_entry)


def setup_structured_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Setup structured logging for better log parsing.
    
    Args:
        level: Logging level
        log_file: Path to log file
    """
    formatter = StructuredFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    _configure_logger_levels()


# Context manager for temporary logging configuration
class TemporaryLoggingConfig:
    """Context manager for temporary logging configuration changes."""
    
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.original_handlers = None
        self.original_level = None
    
    def __enter__(self):
        """Enter the context with new logging configuration."""
        root_logger = logging.getLogger()
        self.original_handlers = root_logger.handlers.copy()
        self.original_level = root_logger.level
        
        # Apply new configuration
        setup_logging(**self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original configuration."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.handlers.extend(self.original_handlers)
        root_logger.setLevel(self.original_level) 