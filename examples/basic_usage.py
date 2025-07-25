#!/usr/bin/env python3
"""
Basic usage example for the RAG-based Image Generation System.

This script demonstrates how to:
1. Set up the system
2. Process a dataset
3. Generate images from text prompts
"""

from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
from src.generation.image_generator import ImageGenerator


def main():
    # Initialize managers
    db_manager = DatabaseEmbeddingManager()
    image_generator = ImageGenerator()

    # Example: Search for similar images
    query = "a cat playing"
    results = db_manager.search_similar(query, k=5)

    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Similarity: {result['similarity_score']:.3f}")

    # Example: Generate an image
    prompt = "a cat playing with a ball"
    generated_image = image_generator.generate(prompt)
    print(f"Generated image: {generated_image}")


if __name__ == "__main__":
    main()
