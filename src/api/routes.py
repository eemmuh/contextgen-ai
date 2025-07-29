"""
API routes for the RAG-based Image Generation System.

This module provides comprehensive API endpoints with proper validation,
error handling, and performance optimizations.
"""

import time
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query, Path, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request

from src.api.schemas import (
    SearchRequest, SearchResponse, SearchResult, ImageMetadata,
    GenerationRequest, GenerationResponse, GenerationResult, GenerationStatus,
    RAGGenerationRequest, RAGGenerationResponse, RAGResult,
    DatabaseStats, HealthCheck, ErrorResponse, SuccessResponse,
    PaginatedResponse, ImageFormat
)
from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
from src.generation.image_generator import ImageGenerator
from src.retrieval.rag_manager import RAGManager
from src.core.exceptions import (
    BaseAppException, ResourceNotFoundError, ValidationError,
    ImageGenerationError, EmbeddingError, DatabaseError
)
from src.core.cache import get_cache_manager, cache_result
from src.utils.logger import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


def get_embedding_manager() -> DatabaseEmbeddingManager:
    """Dependency to get embedding manager instance."""
    return DatabaseEmbeddingManager()


def get_image_generator() -> ImageGenerator:
    """Dependency to get image generator instance."""
    return ImageGenerator()


def get_rag_manager(embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)) -> RAGManager:
    """Dependency to get RAG manager instance."""
    return RAGManager(embedding_manager=embedding_manager)


@router.get("/health", response_model=HealthCheck)
async def health_check(
    request: Request,
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Health check endpoint."""
    start_time = time.time()
    
    try:
        # Check database connection
        db_connected = embedding_manager.check_connection()
        
        # Check model cache health
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        cache_healthy = "error" not in cache_stats.get("memory", {})
        
        uptime = time.time() - start_time
        
        return HealthCheck(
            status="healthy" if db_connected and cache_healthy else "degraded",
            timestamp=time.time(),
            version=settings.api.version,
            uptime_seconds=uptime,
            database_connected=db_connected,
            cache_healthy=cache_healthy
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=time.time(),
            version=settings.api.version,
            uptime_seconds=time.time() - start_time,
            database_connected=False,
            cache_healthy=False
        )


@router.post("/search", response_model=SearchResponse)
@cache_result(ttl=300, key_prefix="search")  # Cache for 5 minutes
async def search_images(
    request: SearchRequest,
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Search for similar images using vector similarity."""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.query.strip():
            raise ValidationError("query", request.query, "Query cannot be empty")
        
        # Perform search
        raw_results = embedding_manager.search_similar(
            query=request.query,
            k=request.k,
            threshold=request.threshold
        )
        
        # Convert to response format
        results = []
        for result in raw_results:
            image_metadata = ImageMetadata(
                id=result["id"],
                filename=result["filename"],
                description=result.get("description"),
                tags=result.get("tags", []),
                width=result.get("width"),
                height=result.get("height"),
                file_size=result.get("file_size"),
                created_at=result["created_at"],
                updated_at=result["updated_at"]
            )
            
            search_result = SearchResult(
                image=image_metadata,
                similarity_score=result["similarity_score"],
                image_url=result.get("image_url")
            )
            results.append(search_result)
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except EmbeddingError as e:
        logger.error(f"Embedding error during search: {e}")
        raise HTTPException(status_code=500, detail="Search failed due to embedding error")
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate", response_model=GenerationResponse)
async def generate_image(
    request: GenerationRequest,
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """Generate images from text prompt."""
    try:
        # Validate input
        if not request.prompt.strip():
            raise ValidationError("prompt", request.prompt, "Prompt cannot be empty")
        
        if request.width % 8 != 0 or request.height % 8 != 0:
            raise ValidationError(
                "width/height", f"{request.width}x{request.height}",
                "Width and height must be divisible by 8"
            )
        
        # Generate images asynchronously
        generation_id = f"gen_{int(time.time())}_{hash(request.prompt) % 10000}"
        
        # Generate images using the actual image generator
        try:
            generated_images = await image_generator.generate_async(
                prompt=request.prompt,
                num_images=request.num_images,
                width=request.width,
                height=request.height
            )
            
            # Save generated images and create URLs
            image_urls = []
            for i, image in enumerate(generated_images):
                # Save image to disk
                image_path = f"output/generated/{generation_id}_{i}.{request.format.value.lower()}"
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)
                
                # Create URL for the saved image
                image_urls.append(f"/api/v1/images/{generation_id}/{i}")
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise ImageGenerationError(f"Failed to generate images: {str(e)}")
        
        generation_result = GenerationResult(
            id=generation_id,
            status=GenerationStatus.COMPLETED,
            prompt=request.prompt,
            enhanced_prompt=None,  # No RAG enhancement in this endpoint
            image_urls=image_urls,
            created_at=time.time(),
            completed_at=time.time()
        )
        
        return GenerationResponse(
            generation=generation_result,
            metadata={
                "width": request.width,
                "height": request.height,
                "format": request.format.value,
                "use_rag": request.use_rag
            }
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except ImageGenerationError as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/rag/generate", response_model=RAGGenerationResponse)
async def generate_with_rag(
    request: RAGGenerationRequest,
    rag_manager: RAGManager = Depends(get_rag_manager),
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """Generate images using RAG techniques."""
    try:
        # Validate input
        if not request.prompt.strip():
            raise ValidationError("prompt", request.prompt, "Prompt cannot be empty")
        
        # Process query through RAG
        rag_output = rag_manager.process_query(
            query=request.prompt,
            k=request.similar_examples_count
        )
        
        # Generate enhanced prompt
        enhanced_prompt = rag_output["augmented_prompt"]
        
        # Generate images
        generation_id = f"rag_gen_{int(time.time())}_{hash(request.prompt) % 10000}"
        
        generated_images = image_generator.generate(
            prompt=enhanced_prompt,
            num_images=request.num_images
        )
        
        # Convert to response format
        image_urls = [f"/api/v1/images/{generation_id}/{i}" for i in range(len(generated_images))]
        
        # Create RAG result
        similar_examples = []
        for example in rag_output["similar_examples"]:
            image_metadata = ImageMetadata(
                id=example["id"],
                filename=example["filename"],
                description=example.get("description"),
                tags=example.get("tags", []),
                width=example.get("width"),
                height=example.get("height"),
                file_size=example.get("file_size"),
                created_at=example["created_at"],
                updated_at=example["updated_at"]
            )
            
            search_result = SearchResult(
                image=image_metadata,
                similarity_score=example["similarity_score"],
                image_url=example.get("image_url")
            )
            similar_examples.append(search_result)
        
        rag_result = RAGResult(
            original_query=request.prompt,
            augmented_query=enhanced_prompt,
            similar_examples=similar_examples,
            confidence_score=rag_output.get("confidence_score", 0.8)
        )
        
        generation_result = GenerationResult(
            id=generation_id,
            status=GenerationStatus.COMPLETED,
            prompt=request.prompt,
            enhanced_prompt=enhanced_prompt,
            image_urls=image_urls,
            created_at=time.time(),
            completed_at=time.time()
        )
        
        return RAGGenerationResponse(
            generation=generation_result,
            rag_result=rag_result,
            metadata={
                "width": request.width,
                "height": request.height,
                "format": request.format.value,
                "similar_examples_count": request.similar_examples_count
            }
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except (ImageGenerationError, EmbeddingError) as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")
    except Exception as e:
        logger.error(f"Unexpected error during RAG generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/upload", response_model=SuccessResponse)
async def upload_image(
    file: UploadFile = File(...),
    description: Optional[str] = Query(None, max_length=1000),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Upload and process an image."""
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise ValidationError("file", file.filename, "File must be an image")
        
        if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
            raise ValidationError("file", file.filename, "File size must be less than 10MB")
        
        # Process tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Process uploaded image
        result = embedding_manager.add_image_with_embeddings(
            image_path=file.filename,
            metadata={
                "description": description,
                "tags": tag_list,
                "uploaded_by": "api",
                "content_type": file.content_type,
                "file_size": file.size
            }
        )
        
        return SuccessResponse(
            message="Image uploaded successfully",
            data={"image_id": result},
            timestamp=time.time()
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


@router.get("/images", response_model=PaginatedResponse)
async def list_images(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """List all images in the database with pagination."""
    try:
        images = embedding_manager.get_all_images(limit=limit, offset=offset)
        total = embedding_manager.get_total_image_count()
        
        # Convert to response format
        image_list = []
        for image in images:
            image_metadata = ImageMetadata(
                id=image["id"],
                filename=image["filename"],
                description=image.get("description"),
                tags=image.get("tags", []),
                width=image.get("width"),
                height=image.get("height"),
                file_size=image.get("file_size"),
                created_at=image["created_at"],
                updated_at=image["updated_at"]
            )
            image_list.append(image_metadata)
        
        pages = (total + limit - 1) // limit
        current_page = (offset // limit) + 1
        
        return PaginatedResponse(
            items=image_list,
            total=total,
            page=current_page,
            per_page=limit,
            pages=pages,
            has_next=current_page < pages,
            has_prev=current_page > 1
        )
        
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail="Failed to list images")


@router.get("/images/{image_id}", response_model=ImageMetadata)
async def get_image(
    image_id: int = Path(..., ge=1),
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Get specific image details."""
    try:
        image = embedding_manager.get_image_by_id(image_id)
        if not image:
            raise ResourceNotFoundError("Image", image_id)
        
        return ImageMetadata(
            id=image["id"],
            filename=image["filename"],
            description=image.get("description"),
            tags=image.get("tags", []),
            width=image.get("width"),
            height=image.get("height"),
            file_size=image.get("file_size"),
            created_at=image["created_at"],
            updated_at=image["updated_at"]
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Error getting image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get image")


@router.delete("/images/{image_id}", response_model=SuccessResponse)
async def delete_image(
    image_id: int = Path(..., ge=1),
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Delete an image from the database."""
    try:
        success = embedding_manager.delete_image(image_id)
        if not success:
            raise ResourceNotFoundError("Image", image_id)
        
        return SuccessResponse(
            message="Image deleted successfully",
            data={"image_id": image_id},
            timestamp=time.time()
        )
        
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.to_dict())
    except Exception as e:
        logger.error(f"Error deleting image {image_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete image")


@router.get("/stats", response_model=DatabaseStats)
@cache_result(ttl=60, key_prefix="stats")  # Cache for 1 minute
async def get_stats(
    embedding_manager: DatabaseEmbeddingManager = Depends(get_embedding_manager)
):
    """Get database statistics."""
    try:
        stats = embedding_manager.get_database_stats()
        
        return DatabaseStats(
            total_images=stats.get("total_images", 0),
            total_embeddings=stats.get("total_embeddings", 0),
            total_generations=stats.get("total_generations", 0),
            storage_size_mb=stats.get("storage_size_mb", 0.0),
            last_updated=time.time()
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@router.get("/rag/search", response_model=RAGResult)
@cache_result(ttl=300, key_prefix="rag_search")  # Cache for 5 minutes
async def search_with_rag(
    query: str = Query(..., min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=20),
    rag_manager: RAGManager = Depends(get_rag_manager)
):
    """Search using RAG techniques."""
    start_time = time.time()
    
    try:
        rag_output = rag_manager.process_query(query=query, k=k)
        
        # Convert to response format
        similar_examples = []
        for example in rag_output["similar_examples"]:
            image_metadata = ImageMetadata(
                id=example["id"],
                filename=example["filename"],
                description=example.get("description"),
                tags=example.get("tags", []),
                width=example.get("width"),
                height=example.get("height"),
                file_size=example.get("file_size"),
                created_at=example["created_at"],
                updated_at=example["updated_at"]
            )
            
            search_result = SearchResult(
                image=image_metadata,
                similarity_score=example["similarity_score"],
                image_url=example.get("image_url")
            )
            similar_examples.append(search_result)
        
        return RAGResult(
            original_query=query,
            augmented_query=rag_output["augmented_prompt"],
            similar_examples=similar_examples,
            confidence_score=rag_output.get("confidence_score", 0.8)
        )
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail="RAG search failed")


# Note: Exception handlers should be registered on the main app, not the router
