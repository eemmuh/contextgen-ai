"""
Comprehensive logging system for the RAG-based Image Generation System.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import traceback

class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to level name
        formatted = formatted.replace(
            record.levelname,
            f"{color}{record.levelname}{reset}"
        )
        
        return formatted

class LoggerManager:
    """Centralized logger management system."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logger manager.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        
        # Configure root logger
        self._configure_root_logger()
        
        # Create loggers for different components
        self.loggers = {}
        self._create_component_loggers()
    
    def _configure_root_logger(self):
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with structured JSON output
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    def _create_component_loggers(self):
        """Create loggers for different system components."""
        components = [
            'cache',
            'embedding',
            'generation',
            'retrieval',
            'dataset',
            'main'
        ]
        
        for component in components:
            logger = logging.getLogger(f"src.{component}")
            self.loggers[component] = logger
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get a logger for a specific component."""
        if component not in self.loggers:
            logger = logging.getLogger(f"src.{component}")
            self.loggers[component] = logger
        return self.loggers[component]
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        logger = self.get_logger('performance')
        extra_fields = {
            'operation': operation,
            'duration_ms': round(duration * 1000, 2),
            'performance_metric': True
        }
        extra_fields.update(kwargs)
        
        logger.info(f"Performance: {operation} completed in {duration:.3f}s", 
                   extra={'extra_fields': extra_fields})
    
    def log_memory_usage(self, component: str, memory_mb: float):
        """Log memory usage for a component."""
        logger = self.get_logger('memory')
        logger.info(f"Memory usage for {component}: {memory_mb:.1f} MB",
                   extra={'extra_fields': {
                       'component': component,
                       'memory_mb': memory_mb,
                       'memory_metric': True
                   }})
    
    def log_cache_event(self, event: str, **kwargs):
        """Log cache-related events."""
        logger = self.get_logger('cache')
        logger.info(f"Cache event: {event}", 
                   extra={'extra_fields': {
                       'cache_event': event,
                       **kwargs
                   }})
    
    def log_error_with_context(self, error: Exception, context: str, **kwargs):
        """Log errors with additional context."""
        logger = self.get_logger('error')
        logger.error(f"Error in {context}: {str(error)}", 
                    exc_info=True,
                    extra={'extra_fields': {
                        'error_context': context,
                        'error_type': type(error).__name__,
                        **kwargs
                    }})

# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None

def get_logger_manager() -> LoggerManager:
    """Get the global logger manager instance."""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager

def get_logger(component: str) -> logging.Logger:
    """Get a logger for a specific component."""
    return get_logger_manager().get_logger(component)

def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics."""
    get_logger_manager().log_performance(operation, duration, **kwargs)

def log_memory_usage(component: str, memory_mb: float):
    """Log memory usage for a component."""
    get_logger_manager().log_memory_usage(component, memory_mb)

def log_cache_event(event: str, **kwargs):
    """Log cache-related events."""
    get_logger_manager().log_cache_event(event, **kwargs)

def log_error_with_context(error: Exception, context: str, **kwargs):
    """Log errors with additional context."""
    get_logger_manager().log_error_with_context(error, context, **kwargs) 