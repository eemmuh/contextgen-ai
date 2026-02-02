"""
API schemas for request and response models.

This module defines Pydantic models for API input validation and response formatting.
"""

import re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, field_validator
from datetime import datetime
from enum import Enum

# Control and other unsafe characters in user text
_CONTROL_AND_UNSAFE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_user_text(v: str) -> str:
    """Strip and remove control/unsafe characters from user text."""
    if not isinstance(v, str):
        return v
    v = v.replace("\x00", "").strip()
    v = _CONTROL_AND_UNSAFE.sub(" ", v)
    return re.sub(r"\s+", " ", v).strip()


class ImageFormat(str, Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class GenerationStatus(str, Enum):
    """Image generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchRequest(BaseModel):
    """Request model for image search."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Similarity threshold")

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        out = _sanitize_user_text(v)
        if not out:
            raise ValueError("Query cannot be empty after sanitization")
        return out


class GenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    num_images: int = Field(default=1, ge=1, le=10, description="Number of images to generate")

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        out = _sanitize_user_text(v)
        if not out:
            raise ValueError("Prompt cannot be empty after sanitization")
        return out
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducibility")
    width: int = Field(default=512, ge=256, le=1024, description="Image width")
    height: int = Field(default=512, ge=256, le=1024, description="Image height")
    format: ImageFormat = Field(default=ImageFormat.PNG, description="Output image format")
    use_rag: bool = Field(default=True, description="Use RAG enhancement")


class RAGGenerationRequest(BaseModel):
    """Request model for RAG-enhanced image generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for generation")
    num_images: int = Field(default=1, ge=1, le=10, description="Number of images to generate")

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        out = _sanitize_user_text(v)
        if not out:
            raise ValueError("Prompt cannot be empty after sanitization")
        return out
    similar_examples_count: int = Field(default=3, ge=1, le=10, description="Number of similar examples to use")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed for reproducibility")
    width: int = Field(default=512, ge=256, le=1024, description="Image width")
    height: int = Field(default=512, ge=256, le=1024, description="Image height")
    format: ImageFormat = Field(default=ImageFormat.PNG, description="Output image format")


class ImageMetadata(BaseModel):
    """Image metadata model."""
    id: int = Field(..., description="Image ID")
    filename: str = Field(..., description="Image filename")
    description: Optional[str] = Field(default=None, description="Image description")
    tags: List[str] = Field(default_factory=list, description="Image tags")
    width: Optional[int] = Field(default=None, description="Image width")
    height: Optional[int] = Field(default=None, description="Image height")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class SearchResult(BaseModel):
    """Search result model."""
    image: ImageMetadata = Field(..., description="Image metadata")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    image_url: Optional[HttpUrl] = Field(default=None, description="Image URL")


class GenerationResult(BaseModel):
    """Image generation result model."""
    id: str = Field(..., description="Generation ID")
    status: GenerationStatus = Field(..., description="Generation status")
    prompt: str = Field(..., description="Original prompt")
    enhanced_prompt: Optional[str] = Field(default=None, description="Enhanced prompt (if RAG used)")
    image_urls: List[HttpUrl] = Field(default_factory=list, description="Generated image URLs")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class RAGResult(BaseModel):
    """RAG processing result model."""
    original_query: str = Field(..., description="Original query")
    augmented_query: str = Field(..., description="Augmented query")
    similar_examples: List[SearchResult] = Field(default_factory=list, description="Similar examples found")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="RAG confidence score")


class DatabaseStats(BaseModel):
    """Database statistics model."""
    total_images: int = Field(..., description="Total number of images")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    total_generations: int = Field(..., description="Total number of generations")
    storage_size_mb: float = Field(..., description="Storage size in MB")
    last_updated: datetime = Field(..., description="Last update timestamp")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    database_connected: bool = Field(..., description="Database connection status")
    cache_healthy: bool = Field(..., description="Model cache health status")
    
    class Config:
        protected_namespaces = ()


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Generic success response model."""
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    timestamp: datetime = Field(..., description="Response timestamp")


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there's a next page")
    has_prev: bool = Field(..., description="Whether there's a previous page")


# Response models for specific endpoints
class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search time in milliseconds")


class GenerationResponse(BaseModel):
    """Response model for generation endpoint."""
    generation: GenerationResult = Field(..., description="Generation result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGGenerationResponse(BaseModel):
    """Response model for RAG generation endpoint."""
    generation: GenerationResult = Field(..., description="Generation result")
    rag_result: RAGResult = Field(..., description="RAG processing result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 