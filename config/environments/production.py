"""
Production environment configuration.

This module contains production-specific settings that override
the default configuration for production deployment.
"""

from config.settings import Settings


class ProductionSettings(Settings):
    """Production environment settings."""
    
    # Environment
    environment: str = "production"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5433/image_rag_db"
    database_echo: bool = False
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Logging settings
    logging_level: str = "INFO"
    logging_file: str = "logs/production.log"
    logging_max_size: int = 50 * 1024 * 1024  # 50MB
    logging_backup_count: int = 10
    
    # Model settings
    model_device: str = "auto"  # Let the system decide
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 5000
    
    # Security settings
    security_access_token_expire_minutes: int = 15  # Shorter tokens for production
    
    class Config:
        env_prefix = "PROD_" 