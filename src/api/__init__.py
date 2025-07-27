"""
API package for the RAG-based Image Generation System.
"""

from .app import create_app
from .routes import router

__all__ = ["create_app", "router"]
