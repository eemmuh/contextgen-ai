#!/usr/bin/env python3
"""
Example usage of the database-backed RAG system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    query = "a dog running"
    results = db_manager.search_similar(query, k=5)
    for result in results:
        print(f"Image: {result['image_path']}")
        print(f"Similarity: {result['similarity_score']:.3f}")

    # Example: Generate an image
    prompt = "a dog running in the park"
    generated_image = image_generator.generate(prompt)
    print(f"Generated image: {generated_image}")


if __name__ == "__main__":
    main()
