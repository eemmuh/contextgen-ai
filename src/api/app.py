"""
FastAPI application for the RAG-based Image Generation System.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.utils.logger import get_logger
from src.utils.error_handler import get_error_handler

logger = get_logger(__name__)
error_handler = get_error_handler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting application...")
    yield
    logger.info("Shutting down application...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG-based Image Generation API",
        description="API for generating images using RAG techniques",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Include routers
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


    @app.get("/")
    async def root():
        """Root endpoint."""
    return {"message": "RAG-based Image Generation API"}


    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/docs")
async def docs():
    """API documentation endpoint."""
    return {"docs_url": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
