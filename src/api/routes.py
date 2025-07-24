"""
API routes for the RAG-based Image Generation System.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import uuid
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger("api_routes")

api_router = APIRouter()


@api_router.get("/images")
async def list_images(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by")
):
    """List images in the database."""
    try:
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        db_manager = request.app.state.db_manager
        
        # Parse tags filter
        tag_list = tags.split(",") if tags else None
        
        # Get images (this would need to be implemented in DatabaseManager)
        # For now, return a placeholder
        return {
            "images": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail="Failed to list images")


@api_router.post("/images/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Upload and process an image."""
    try:
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4()}{file_extension}"
        file_path = f"uploads/{filename}"
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Parse tags
        tag_list = tags.split(",") if tags else []
        
        # Add to database
        db_manager = request.app.state.db_manager
        
        image_id = db_manager.add_image(
            image_path=file_path,
            description=description,
            tags=tag_list
        )
        
        return {
            "message": "Image uploaded successfully",
            "image_id": image_id,
            "file_path": file_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")


@api_router.get("/search")
async def search_images(
    request: Request,
    query: str = Query(..., description="Text query for image search"),
    k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    model_type: str = Query("sentence_transformer", description="Embedding model type")
):
    """Search for similar images using text query."""
    try:
        if not request.app.state.embedding_manager:
            raise HTTPException(status_code=503, detail="Embedding manager not available")
        
        embedding_manager = request.app.state.embedding_manager
        
        # Search for similar images
        results = embedding_manager.search_similar(
            query=query,
            k=k,
            model_type=model_type
        )
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching images: {e}")
        raise HTTPException(status_code=500, detail="Failed to search images")


@api_router.post("/generate")
async def generate_image(
    request: Request,
    prompt: str = Form(..., description="Text prompt for image generation"),
    seed: Optional[int] = Form(None, description="Random seed for generation"),
    num_inference_steps: int = Form(50, ge=1, le=100),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0)
):
    """Generate an image using RAG-enhanced prompt."""
    try:
        if not request.app.state.rag_manager:
            raise HTTPException(status_code=503, detail="RAG manager not available")
        if not request.app.state.image_generator:
            raise HTTPException(status_code=503, detail="Image generator not available")
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        rag_manager = request.app.state.rag_manager
        image_generator = request.app.state.image_generator
        db_manager = request.app.state.db_manager
        
        # Augment prompt with RAG
        augmented_prompt = rag_manager.augment_prompt(prompt)
        
        # Generate image
        output_path = image_generator.generate(
            prompt=augmented_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Record generation
        generation_id = db_manager.add_generation(
            prompt=prompt,
            augmented_prompt=augmented_prompt,
            output_path=output_path,
            seed=seed
        )
        
        return {
            "message": "Image generated successfully",
            "generation_id": generation_id,
            "original_prompt": prompt,
            "augmented_prompt": augmented_prompt,
            "output_path": output_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image")


@api_router.get("/generations")
async def list_generations(
    request: Request,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List image generations."""
    try:
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        db_manager = request.app.state.db_manager
        
        history = db_manager.get_generation_history(
            limit=limit,
            status=status
        )
        
        return {
            "generations": history,
            "total": len(history),
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing generations: {e}")
        raise HTTPException(status_code=500, detail="Failed to list generations")


@api_router.get("/stats")
async def get_stats(request: Request):
    """Get system statistics."""
    try:
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        db_manager = request.app.state.db_manager
        
        stats = db_manager.get_database_stats()
        
        return {
            "database_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


@api_router.get("/images/{image_id}")
async def get_image(request: Request, image_id: int):
    """Get image details by ID."""
    try:
        if not request.app.state.db_manager:
            raise HTTPException(status_code=503, detail="Database manager not available")
        
        db_manager = request.app.state.db_manager
        
        image = db_manager.get_image_by_id(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "image": {
                "id": image.id,
                "image_path": image.image_path,
                "description": image.description,
                "tags": image.tags,
                "width": image.width,
                "height": image.height,
                "created_at": image.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {e}")
        raise HTTPException(status_code=500, detail="Failed to get image")


@api_router.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve static files (images, generated content)."""
    try:
        full_path = os.path.join("uploads", file_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(full_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve file")


@api_router.get("/status")
async def api_status(request: Request):
    """Get API component status."""
    components = {}
    
    # Check database manager
    if request.app.state.db_manager:
        try:
            stats = request.app.state.db_manager.get_database_stats()
            components["database"] = {"status": "available", "stats": stats}
        except Exception as e:
            components["database"] = {"status": "error", "error": str(e)}
    else:
        components["database"] = {"status": "not_available"}
    
    # Check embedding manager
    components["embedding_manager"] = {
        "status": "available" if request.app.state.embedding_manager else "not_available"
    }
    
    # Check image generator
    components["image_generator"] = {
        "status": "available" if request.app.state.image_generator else "not_available"
    }
    
    # Check RAG manager
    components["rag_manager"] = {
        "status": "available" if request.app.state.rag_manager else "not_available"
    }
    
    return {
        "api_status": "running",
        "components": components,
        "timestamp": datetime.now().isoformat()
    } 