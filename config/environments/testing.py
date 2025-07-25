"""
Testing environment configuration.

This module contains testing-specific settings that override
the default configuration for automated testing.
"""

from config.settings import Settings


class TestingSettings(Settings):
    """Testing environment settings."""
    
    # Environment
    environment: str = "testing"
    debug: bool = False
    
    # API settings
    api_host: str = "127.0.0.1"
    api_port: int = 8001
    api_debug: bool = False
    
    # Database settings
    database_url: str = "postgresql://postgres:password@localhost:5433/image_rag_db_test"
    database_echo: bool = False
    
    # Logging settings
    logging_level: str = "WARNING"
    logging_file: str = "logs/testing.log"
    
    # Model settings
    model_device: str = "cpu"
    model_cache_dir: str = ".model_cache_test"
    
    # Cache settings
    cache_enabled: bool = False  # Disable caching for tests
    
    class Config:
        env_prefix = "TEST_" 