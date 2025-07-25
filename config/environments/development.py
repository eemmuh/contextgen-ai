"""
Development environment configuration.

This module contains development-specific settings that override
the default configuration for local development.
"""

from config.settings import Settings


class DevelopmentSettings(Settings):
    """Development environment settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_debug: bool = True
    
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5433/image_rag_db_dev"
    database_echo: bool = True
    
    # Logging settings
    logging_level: str = "DEBUG"
    logging_file: str = "logs/development.log"
    
    # Model settings
    model_device: str = "cpu"  # Use CPU for development to avoid GPU issues
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes for development
    
    class Config:
        env_prefix = "DEV_" 