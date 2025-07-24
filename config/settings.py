"""
Settings management using Pydantic for the RAG-based Image Generation System.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://postgres:password@localhost:5433/image_rag_db",
        description="Database connection URL"
    )
    pool_size: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Maximum overflow connections")
    pool_recycle: int = Field(default=3600, description="Connection pool recycle time in seconds")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    
    class Config:
        env_prefix = "DB_"


class ModelSettings(BaseSettings):
    """Model configuration settings."""
    
    text_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Text embedding model name"
    )
    image_model: str = Field(
        default="openai/clip-vit-base-patch32",
        description="Image embedding model name"
    )
    cache_dir: str = Field(
        default=".model_cache",
        description="Model cache directory"
    )
    device: str = Field(
        default="auto",
        description="Device to use for models (auto, cpu, cuda)"
    )
    
    class Config:
        env_prefix = "MODEL_"


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    max_memory_size_mb: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Maximum memory cache size in MB"
    )
    max_disk_size_mb: int = Field(
        default=10240,
        ge=1024,
        le=51200,
        description="Maximum disk cache size in MB"
    )
    compression_enabled: bool = Field(
        default=True,
        description="Enable cache compression"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable model validation"
    )
    
    class Config:
        env_prefix = "CACHE_"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    reload: bool = Field(default=False, description="Enable auto-reload for development")
    workers: int = Field(default=1, ge=1, le=10, description="Number of worker processes")
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    max_size_mb: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files"
    )
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class VectorSearchSettings(BaseSettings):
    """Vector search configuration settings."""
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for search results"
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of search results"
    )
    index_type: str = Field(
        default="ivfflat",
        description="Vector index type (ivfflat, hnsw)"
    )
    
    class Config:
        env_prefix = "VECTOR_"


class GenerationSettings(BaseSettings):
    """Image generation configuration settings."""
    
    default_steps: int = Field(
        default=50,
        ge=10,
        le=100,
        description="Default number of inference steps"
    )
    default_guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="Default guidance scale"
    )
    output_dir: str = Field(
        default="output",
        description="Output directory for generated images"
    )
    image_size: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Default image size"
    )
    
    class Config:
        env_prefix = "GEN_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    vector_search: VectorSearchSettings = Field(default_factory=VectorSearchSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    
    # Paths
    base_dir: Path = Field(default=Path.cwd(), description="Base directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    uploads_dir: Path = Field(default=Path("uploads"), description="Uploads directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    
    @validator('base_dir', 'data_dir', 'uploads_dir', 'logs_dir', pre=True)
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings 