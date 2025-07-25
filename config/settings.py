"""
Application settings and configuration management.

This module provides centralized configuration management for the application,
supporting different environments (development, testing, production) and
secure handling of sensitive information.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from pydantic.types import SecretStr


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://postgres:password@localhost:5433/image_rag_db",
        description="Database connection URL"
    )
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    
    class Config:
        env_prefix = "DB_"


class ModelSettings(BaseSettings):
    """Machine learning model configuration settings."""
    
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
        description="Device to use for model inference (cpu, cuda, auto)"
    )
    
    class Config:
        env_prefix = "MODEL_"


class APISettings(BaseSettings):
    """API configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    title: str = Field(default="Image Model COCO API", description="API title")
    version: str = Field(default="0.1.0", description="API version")
    docs_url: str = Field(default="/docs", description="API documentation URL")
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file: Optional[str] = Field(default="logs/app.log", description="Log file path")
    max_size: int = Field(default=10 * 1024 * 1024, description="Max log file size (bytes)")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    class Config:
        env_prefix = "LOG_"


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    secret_key: SecretStr = Field(
        default=SecretStr("your-secret-key-here"),
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, description="Maximum cache size")
    
    class Config:
        env_prefix = "CACHE_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Project paths
    base_dir: Path = Field(default=Path(__file__).parent.parent, description="Base directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    models_dir: Path = Field(default=Path("data/models"), description="Models directory")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    
    # External services
    pexels_api_key: Optional[SecretStr] = Field(default=None, description="Pexels API key")
    
    @validator("base_dir", "data_dir", "logs_dir", "models_dir", pre=True)
    def validate_paths(cls, v):
        """Ensure paths are Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_environments = ["development", "testing", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        return self.database.url
    
    def get_model_cache_dir(self) -> Path:
        """Get model cache directory path."""
        return self.base_dir / self.model.cache_dir
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get log file path."""
        if self.logging.file:
            return self.logs_dir / self.logging.file
        return None
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == "testing"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload application settings."""
    global _settings
    _settings = Settings()
    return _settings


# Convenience function for getting settings
settings = get_settings() 