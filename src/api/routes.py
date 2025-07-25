"""
API routes for the RAG-based Image Generation System.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
from src.generation.image_generator import ImageGenerator
from src.retrieval.rag_manager import RAGManager

router = APIRouter()


@router.get("/search")
async def search_images(query: str, k: int = 5):
    """Search for similar images."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        results = embedding_manager.search_similar(query, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_image(prompt: str, num_images: int = 1):
    """Generate images from text prompt."""
    try:
        image_generator = ImageGenerator()
        results = image_generator.generate(prompt, num_images=num_images)
        return {"generated_images": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process an image."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        # Process uploaded image
        result = embedding_manager.add_image_with_embeddings(
            image_path=file.filename,
            metadata={"uploaded_by": "api"},
        )
        return {"message": "Image uploaded successfully", "image_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images")
async def list_images(limit: int = 10, offset: int = 0):
    """List all images in the database."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        images = embedding_manager.get_all_images(limit=limit, offset=offset)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images/{image_id}")
async def get_image(image_id: int):
    """Get specific image details."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        image = embedding_manager.get_image_by_id(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"image": image}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/images/{image_id}")
async def delete_image(image_id: int):
    """Delete an image from the database."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        success = embedding_manager.delete_image(image_id)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"message": "Image deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get database statistics."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        stats = embedding_manager.get_database_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/generate")
async def generate_with_rag(
    prompt: str,
    num_images: int = 1,
    use_similar_examples: bool = True,
):
    """Generate images using RAG techniques."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        rag_manager = RAGManager(embedding_manager=embedding_manager)
        image_generator = ImageGenerator()

        # Process query through RAG
        rag_output = rag_manager.process_query(prompt)

        # Generate image with enhanced prompt
        if use_similar_examples and rag_output["similar_examples"]:
            enhanced_prompt = rag_output["augmented_prompt"]
        else:
            enhanced_prompt = prompt

        results = image_generator.generate(enhanced_prompt, num_images=num_images)

        return {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "similar_examples": rag_output["similar_examples"],
            "generated_images": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/search")
async def search_with_rag(query: str, k: int = 5):
    """Search using RAG techniques."""
    try:
        embedding_manager = DatabaseEmbeddingManager()
        rag_manager = RAGManager(embedding_manager=embedding_manager)

        rag_output = rag_manager.process_query(query)

        return {
            "original_query": query,
            "augmented_query": rag_output["augmented_prompt"],
            "similar_examples": rag_output["similar_examples"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
