#!/usr/bin/env python3
"""
Example usage of the database-backed RAG system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
from src.database.database import DatabaseManager
from src.database.session import test_connection, create_tables
from src.utils.logger import get_logger

logger = get_logger("database_example")


def setup_database():
    """Setup database if not already configured."""
    if not test_connection():
        logger.error("Database connection failed. Please run 'make setup-db' first.")
        return False
    
    try:
        create_tables()
        logger.info("Database tables created/verified")
        return True
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        return False


def example_basic_usage():
    """Example of basic database operations."""
    logger.info("=== Basic Database Usage Example ===")
    
    # Initialize managers
    db_manager = DatabaseManager()
    embedding_manager = DatabaseEmbeddingManager()
    
    # Add some sample images
    sample_images = [
        {
            "path": "data/sample/cat.jpg",
            "metadata": {
                "description": "A cute cat sitting on a chair",
                "tags": ["cat", "chair", "cute", "pet"],
                "width": 800,
                "height": 600,
                "source_dataset": "sample"
            }
        },
        {
            "path": "data/sample/dog.jpg", 
            "metadata": {
                "description": "A happy dog playing in the park",
                "tags": ["dog", "park", "happy", "playing"],
                "width": 1024,
                "height": 768,
                "source_dataset": "sample"
            }
        },
        {
            "path": "data/sample/landscape.jpg",
            "metadata": {
                "description": "Beautiful mountain landscape with trees",
                "tags": ["mountain", "landscape", "trees", "nature"],
                "width": 1920,
                "height": 1080,
                "source_dataset": "sample"
            }
        }
    ]
    
    # Add images to database (without actual files for demo)
    for i, img in enumerate(sample_images):
        try:
            image_id = embedding_manager.add_image_with_embeddings(
                image_path=img["path"],
                metadata=img["metadata"],
                compute_embeddings=True
            )
            logger.info(f"Added image {i+1}: {img['path']} (ID: {image_id})")
        except Exception as e:
            logger.warning(f"Could not add image {img['path']}: {e}")
    
    # Search for similar images
    queries = [
        "a cat playing",
        "outdoor activities", 
        "nature scenes",
        "pets and animals"
    ]
    
    for query in queries:
        logger.info(f"\nSearching for: '{query}'")
        results = embedding_manager.search_similar(
            query=query,
            k=3,
            similarity_threshold=0.3
        )
        
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. {result['description']} (score: {result['similarity_score']:.3f})")
    
    # Get database statistics
    stats = embedding_manager.get_database_stats()
    logger.info(f"\nDatabase Statistics:")
    logger.info(f"  Total images: {stats['total_images']}")
    logger.info(f"  Total embeddings: {stats['total_embeddings']}")
    logger.info(f"  Total generations: {stats['total_generations']}")


def example_generation_tracking():
    """Example of tracking image generations."""
    logger.info("\n=== Generation Tracking Example ===")
    
    db_manager = DatabaseManager()
    
    # Simulate some image generations
    sample_generations = [
        {
            "prompt": "a cat sitting on a chair",
            "augmented_prompt": "a cat sitting on a chair, high quality, detailed",
            "output_path": "output/generated_cat_1.png",
            "seed": 42,
            "generation_time_ms": 2500,
            "memory_usage_mb": 512.5,
            "model_config": {"model": "stable-diffusion-v1-5", "steps": 50},
            "retrieved_examples": [{"image_id": 1, "similarity": 0.85}]
        },
        {
            "prompt": "a dog in the park",
            "augmented_prompt": "a happy dog playing in a sunny park, vibrant colors",
            "output_path": "output/generated_dog_1.png", 
            "seed": 123,
            "generation_time_ms": 3200,
            "memory_usage_mb": 498.2,
            "model_config": {"model": "stable-diffusion-v1-5", "steps": 50},
            "retrieved_examples": [{"image_id": 2, "similarity": 0.92}]
        }
    ]
    
    for gen in sample_generations:
        generation_id = db_manager.add_generation(**gen)
        logger.info(f"Tracked generation: {generation_id} - '{gen['prompt'][:30]}...'")
    
    # Get generation history
    recent_generations = db_manager.get_generation_history(limit=5)
    logger.info(f"\nRecent generations: {len(recent_generations)}")
    for gen in recent_generations:
        logger.info(f"  {gen['id']}: {gen['prompt'][:40]}... ({gen['status']})")


def example_system_monitoring():
    """Example of system metrics tracking."""
    logger.info("\n=== System Monitoring Example ===")
    
    db_manager = DatabaseManager()
    
    # Add some sample system metrics
    sample_metrics = [
        {
            "cpu_percent": 45.2,
            "memory_mb": 2048.5,
            "gpu_memory_mb": 3072.0,
            "gpu_utilization": 78.5,
            "disk_usage_percent": 65.3,
            "cache_hit_rate": 85.2,
            "component": "embedding"
        },
        {
            "cpu_percent": 62.1,
            "memory_mb": 3072.8,
            "gpu_memory_mb": 4096.0,
            "gpu_utilization": 92.3,
            "disk_usage_percent": 65.3,
            "cache_hit_rate": 78.9,
            "component": "generation"
        }
    ]
    
    for metrics in sample_metrics:
        db_manager.add_system_metrics(**metrics)
    
    logger.info("Added sample system metrics")


def main():
    """Run the database usage examples."""
    logger.info("Starting Database Usage Examples")
    
    # Setup database
    if not setup_database():
        logger.error("Database setup failed. Exiting.")
        return 1
    
    try:
        # Run examples
        example_basic_usage()
        example_generation_tracking()
        example_system_monitoring()
        
        logger.info("\n✅ All examples completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 