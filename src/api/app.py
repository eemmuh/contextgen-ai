"""
FastAPI application for the RAG-based Image Generation System.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import time
import os

from src.api.routes import router
from src.api.middleware.rate_limiter import rate_limit_middleware
from src.core.exceptions import BaseAppException
from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger("api")
settings = get_settings()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Image Model COCO API",
        description="""
        **RAG-based Image Generation System with PostgreSQL Vector Database**
        
        A comprehensive Retrieval-Augmented Generation (RAG) system for image generation 
        using the COCO dataset, featuring PostgreSQL database integration with pgvector 
        for efficient vector similarity search.
        
        ## Features
        
        - **Vector Similarity Search**: Fast image retrieval using pgvector
        - **RAG Pipeline**: Context-aware image generation
        - **PostgreSQL Database**: Scalable data storage with vector support
        - **Performance Monitoring**: Comprehensive metrics and logging
        - **COCO Integration**: Full dataset support with metadata
        - **Model Caching**: Intelligent caching for improved performance
        - **Docker Support**: Easy deployment with Docker Compose
        
        ## Quick Start
        
        1. **Health Check**: `GET /health`
        2. **Search Images**: `POST /api/v1/search`
        3. **Generate Images**: `POST /api/v1/generate`
        4. **RAG Generation**: `POST /api/v1/rag/generate`
        
        ## Authentication
        
        Currently, this API does not require authentication. Rate limiting is applied 
        to prevent abuse (60 requests per minute per client).
        
        ## Rate Limiting
        
        - **Limit**: 60 requests per minute per client
        - **Headers**: `X-RateLimit-Remaining`, `X-RateLimit-Limit`
        - **Response**: 429 status code when limit exceeded
        """,
        version="0.1.0",
        contact={
            "name": "API Support",
            "url": "https://github.com/eemmuh/contextgen-ai/issues",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add custom middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header to responses."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    @app.middleware("http")
    async def rate_limiting_middleware(request: Request, call_next):
        """Apply rate limiting to all requests."""
        return await rate_limit_middleware(request, call_next)
    
    # Add exception handlers
    @app.exception_handler(BaseAppException)
    async def app_exception_handler(request: Request, exc: BaseAppException):
        """Handle application-specific exceptions."""
        logger.error(f"Application error: {exc}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": str(exc),
                "details": exc.details
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": str(exc) if settings.debug else "Internal server error"
            }
        )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add rate limiting info
        openapi_schema["info"]["x-rate-limit"] = {
            "requests_per_minute": 60,
            "description": "Rate limiting is applied per client IP"
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Image Model COCO API",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "features": [
                "Vector Similarity Search",
                "RAG Pipeline",
                "PostgreSQL Database",
                "Performance Monitoring",
                "COCO Integration",
                "Model Caching",
                "Docker Support"
            ]
        }
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        try:
            # Add basic health checks here
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0",
                "services": {
                    "api": "healthy",
                    "database": "healthy",  # Add actual DB check
                    "cache": "healthy",     # Add actual cache check
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": time.time()
                }
            )
    
    return app


# Create app instance
app = create_app()
