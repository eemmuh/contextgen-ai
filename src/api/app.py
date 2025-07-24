"""
FastAPI application for the RAG-based Image Generation System.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager
import os

from src.utils.logger import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting API server...")
    yield
    # Shutdown
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Image Model COCO API",
        description="RAG-based Image Generation System with PostgreSQL Vector Database",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components (with error handling)
    try:
        from src.database.database import DatabaseManager
        app.state.db_manager = DatabaseManager()
        logger.info("Database manager initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize database manager: {e}")
        app.state.db_manager = None
    
    try:
        from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
        app.state.embedding_manager = DatabaseEmbeddingManager()
        logger.info("Embedding manager initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize embedding manager: {e}")
        app.state.embedding_manager = None
    
    try:
        from src.generation.image_generator import ImageGenerator
        app.state.image_generator = ImageGenerator()
        logger.info("Image generator initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize image generator: {e}")
        app.state.image_generator = None
    
    try:
        from src.retrieval.rag_manager import RAGManager
        if app.state.embedding_manager:
            app.state.rag_manager = RAGManager(embedding_manager=app.state.embedding_manager)
            logger.info("RAG manager initialized successfully")
        else:
            logger.warning("RAG manager not initialized: embedding manager not available")
            app.state.rag_manager = None
    except Exception as e:
        logger.warning(f"Failed to initialize RAG manager: {e}")
        app.state.rag_manager = None
    
    # Include API routes
    try:
        from .routes import api_router
        app.include_router(api_router, prefix="/api/v1")
        logger.info("API routes loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load API routes: {e}")
        # Create a basic router if the main one fails
        from fastapi import APIRouter
        basic_router = APIRouter()
        
        @basic_router.get("/status")
        async def status():
            return {"status": "API routes not available", "error": str(e)}
        
        app.include_router(basic_router, prefix="/api/v1")
    
    # Mount static files
    try:
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        app.mount("/static", StaticFiles(directory="static"), name="static")
        logger.info("Static files mounted successfully")
    except Exception as e:
        logger.warning(f"Failed to mount static files: {e}")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Image Model COCO API",
            "version": "0.1.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            if app.state.db_manager:
                # Test database connection
                stats = app.state.db_manager.get_database_stats()
                return {
                    "status": "healthy",
                    "database": "connected",
                    "stats": stats
                }
            else:
                return {
                    "status": "degraded",
                    "database": "not_available",
                    "message": "Database manager not initialized"
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "database": "error",
                "error": str(e)
            }
    
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server() 