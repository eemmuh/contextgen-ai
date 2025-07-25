#!/usr/bin/env python3
"""
Advanced Model Caching Demo

This script demonstrates the enhanced model caching functionality:
1. First run: Models are downloaded and cached
2. Second run: Models are loaded from cache (much faster)
3. Advanced cache management features
4. Cache monitoring and optimization
5. Cache warmup capabilities
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.embedding.database_embedding_manager import (
        DatabaseEmbeddingManager
    )
    from src.generation.image_generator import ImageGenerator
except ImportError:
    print(
        "Error: Could not import required modules. "
        "Make sure you're running from the project root."
    )
    sys.exit(1)


def main():
    db_manager = DatabaseEmbeddingManager()
    image_generator = ImageGenerator()

    # Example: Search for similar images
    query = "a bird flying"
    results = db_manager.search_similar(query, k=3)
    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Similarity: {result['similarity_score']:.3f}")

    # Example: Generate an image
    prompt = "a bird flying over the sea"
    generated_image = image_generator.generate(prompt)
    print(f"Generated image: {generated_image}")


if __name__ == "__main__":
    main()
