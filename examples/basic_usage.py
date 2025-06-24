#!/usr/bin/env python3
"""
Basic usage example for the RAG-based Image Generation System.

This script demonstrates how to:
1. Set up the system
2. Process a dataset  
3. Generate images from text prompts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedding_manager import EmbeddingManager
from src.retrieval.rag_manager import RAGManager
from src.generation.image_generator import ImageGenerator
from src.data_processing.coco_dataset import COCODataset

def main():
    print("🎨 RAG-based Image Generation - Basic Usage Example")
    print("=" * 50)
    
    # 1. Initialize components
    print("1. Initializing components...")
    embedding_manager = EmbeddingManager()
    rag_manager = RAGManager(embedding_manager)
    image_generator = ImageGenerator()
    
    # 2. Load pre-processed embeddings (assuming they exist)
    try:
        print("2. Loading existing embeddings...")
        embedding_manager.load_index("embeddings/coco_dataset")
        print("   ✅ Embeddings loaded successfully!")
    except FileNotFoundError:
        print("   ❌ No embeddings found. Please run dataset processing first:")
        print("   python -m src.main --process-dataset --dataset-type coco")
        return
    
    # 3. Generate images from prompts
    test_prompts = [
        "a cat sitting on a chair",
        "a dog playing in a park",
        "a sunset over mountains",
        "a flower in a vase"
    ]
    
    print("3. Generating images from prompts...")
    for i, prompt in enumerate(test_prompts):
        print(f"\n   Processing: '{prompt}'")
        
        # Process query through RAG
        rag_output = rag_manager.process_query(prompt)
        
        print(f"   Original prompt: {rag_output['original_query']}")
        print(f"   Enhanced prompt: {rag_output['augmented_prompt']}")
        print(f"   Found {len(rag_output['similar_examples'])} similar examples")
        
        # Generate image
        result = image_generator.generate_from_rag_output(
            rag_output=rag_output,
            output_dir="examples/output",
            num_images=1,
            seed=42 + i  # For reproducibility
        )
        
        print(f"   ✅ Generated: {result['generated_images'][0]}")
    
    print(f"\n🎉 Generated {len(test_prompts)} images!")
    print("📁 Check the 'examples/output/' directory for results.")

if __name__ == "__main__":
    main() 