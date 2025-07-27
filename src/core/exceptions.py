"""
Custom exceptions for the application.

This module defines a hierarchy of custom exceptions for different types of errors
that can occur in the application, providing better error handling and user feedback.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class BaseAppException(Exception):
    """Base exception for all application errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat()
        }


class ConfigurationError(BaseAppException):
    """Raised when there's a configuration error."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "CONFIG_ERROR", details, 500)


class DatabaseError(BaseAppException):
    """Base exception for database-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DATABASE_ERROR", details, 500)


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "DATABASE_CONNECTION_ERROR"
        self.status_code = 503


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "DATABASE_QUERY_ERROR"
        self.query = query
        if query:
            self.details["query"] = query


class ModelError(BaseAppException):
    """Base exception for model-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MODEL_ERROR", details, 500)


class ModelNotFoundError(ModelError):
    """Raised when a model is not found."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Model '{model_name}' not found"
        super().__init__(message, details)
        self.error_code = "MODEL_NOT_FOUND"
        self.model_name = model_name
        self.details["model_name"] = model_name


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(message, details)
        self.error_code = "MODEL_LOAD_ERROR"
        self.model_name = model_name
        self.reason = reason
        self.details.update({
            "model_name": model_name,
            "reason": reason
        })


class EmbeddingError(BaseAppException):
    """Base exception for embedding-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EMBEDDING_ERROR", details, 500)


class EmbeddingGenerationError(EmbeddingError):
    """Raised when embedding generation fails."""
    
    def __init__(self, text: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to generate embedding for text: {reason}"
        super().__init__(message, details)
        self.error_code = "EMBEDDING_GENERATION_ERROR"
        self.text = text
        self.reason = reason
        self.details.update({
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "reason": reason
        })


class ImageGenerationError(BaseAppException):
    """Base exception for image generation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "IMAGE_GENERATION_ERROR", details, 500)


class InvalidPromptError(ImageGenerationError):
    """Raised when the prompt is invalid."""
    
    def __init__(self, prompt: str, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Invalid prompt: {reason}"
        super().__init__(message, details)
        self.error_code = "INVALID_PROMPT"
        self.prompt = prompt
        self.reason = reason
        self.status_code = 400
        self.details.update({
            "prompt": prompt,
            "reason": reason
        })


class GenerationTimeoutError(ImageGenerationError):
    """Raised when image generation times out."""
    
    def __init__(self, prompt: str, timeout_seconds: int, details: Optional[Dict[str, Any]] = None):
        message = f"Image generation timed out after {timeout_seconds} seconds"
        super().__init__(message, details)
        self.error_code = "GENERATION_TIMEOUT"
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds
        self.status_code = 408
        self.details.update({
            "prompt": prompt,
            "timeout_seconds": timeout_seconds
        })


class ValidationError(BaseAppException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str, details: Optional[Dict[str, Any]] = None):
        message = f"Validation error for field '{field}': {reason}"
        super().__init__(message, details)
        self.error_code = "VALIDATION_ERROR"
        self.field = field
        self.value = value
        self.reason = reason
        self.status_code = 400
        self.details.update({
            "field": field,
            "value": str(value),
            "reason": reason
        })


class ResourceNotFoundError(BaseAppException):
    """Raised when a requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: Any, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} with id '{resource_id}' not found"
        super().__init__(message, details)
        self.error_code = "RESOURCE_NOT_FOUND"
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.status_code = 404
        self.details.update({
            "resource_type": resource_type,
            "resource_id": str(resource_id)
        })


class RateLimitError(BaseAppException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, details: Optional[Dict[str, Any]] = None):
        message = f"Rate limit exceeded: {limit} requests per {window_seconds} seconds"
        super().__init__(message, details)
        self.error_code = "RATE_LIMIT_EXCEEDED"
        self.limit = limit
        self.window_seconds = window_seconds
        self.status_code = 429
        self.details.update({
            "limit": limit,
            "window_seconds": window_seconds
        })


class AuthenticationError(BaseAppException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "AUTHENTICATION_ERROR"
        self.status_code = 401


class AuthorizationError(BaseAppException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "AUTHORIZATION_ERROR"
        self.status_code = 403


class ExternalServiceError(BaseAppException):
    """Raised when an external service call fails."""
    
    def __init__(self, service_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "EXTERNAL_SERVICE_ERROR"
        self.service_name = service_name
        self.status_code = 502
        self.details["service_name"] = service_name


class CacheError(BaseAppException):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.error_code = "CACHE_ERROR"
        self.status_code = 500


# Convenience functions for common error scenarios
def raise_validation_error(field: str, value: Any, reason: str) -> None:
    """Raise a validation error with the given details."""
    raise ValidationError(field, value, reason)


def raise_not_found_error(resource_type: str, resource_id: Any) -> None:
    """Raise a resource not found error."""
    raise ResourceNotFoundError(resource_type, resource_id)


def raise_database_error(message: str, query: Optional[str] = None) -> None:
    """Raise a database error with optional query details."""
    details = {"query": query} if query else None
    raise DatabaseQueryError(message, query, details) 