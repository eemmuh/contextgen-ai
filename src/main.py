import argparse
import os
from dotenv import load_dotenv
from src.data_processing.dataset import ImageMetadataDataset
from src.data_processing.coco_dataset import COCODataset
from src.embedding.embedding_manager import EmbeddingManager
from src.retrieval.rag_manager import RAGManager
from src.generation.image_generator import ImageGenerator
from typing import Optional

def setup_environment():
    """Load environment variables and create necessary directories."""
    load_dotenv()
    
    # Create necessary directories
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

def process_dataset(
    dataset_type: str,
    dataset_path: str,
    metadata_path: Optional[str] = None,
    embedding_manager: Optional[EmbeddingManager] = None,
    max_images: Optional[int] = None
):
    """Process the dataset and create embeddings."""
    print(f"Processing {dataset_type} dataset...")
    
    # Initialize dataset
    if dataset_type == "coco":
        dataset = COCODataset(
            coco_dir=dataset_path,
            max_images=max_images
        )
    else:
        dataset = ImageMetadataDataset(
            image_dir=dataset_path,
            metadata_path=metadata_path,
            max_images=max_images
        )
    
    # Initialize embedding manager if not provided
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
    
    # Process each item in the dataset
    print(f"📊 Processing {len(dataset)} images...")
    
    for i in range(len(dataset)):
        if i % 20 == 0:  # Progress indicator
            print(f"   Processing image {i+1}/{len(dataset)}")
        
        item = dataset[i]
        
        # Compute embeddings
        image_embedding = embedding_manager.compute_image_embedding(item['image'])
        metadata_text = dataset.get_metadata_text(i)
        text_embedding = embedding_manager.compute_text_embedding(metadata_text)
        
        # Add text embedding to index for text-based search
        embedding_manager.add_to_index(
            embedding=text_embedding,  # Use text embedding for text search
            metadata=item['metadata'],
            image_path=item['image_path']
        )
    
    # Save the index
    embedding_manager.save_index(f"embeddings/{dataset_type}_dataset")
    print("Dataset processing complete!")

def main():
    parser = argparse.ArgumentParser(description="RAG-based Image Generation System")
    parser.add_argument("--prompt", type=str, help="Input prompt for image generation")
    parser.add_argument("--dataset-type", type=str, choices=["coco", "custom"], default="coco",
                      help="Type of dataset to use")
    parser.add_argument("--dataset", type=str, default="data/coco",
                      help="Path to dataset directory")
    parser.add_argument("--metadata", type=str, help="Path to metadata CSV (for custom dataset)")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--fast-mode", action="store_true", help="Use fast generation mode (10 steps)")
    parser.add_argument("--process-dataset", action="store_true",
                      help="Process the dataset and create embeddings")
    parser.add_argument("--max-images", type=int, help="Maximum number of images to process")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Initialize components
    embedding_manager = EmbeddingManager()
    
    # Process dataset if requested
    if args.process_dataset:
        process_dataset(
            dataset_type=args.dataset_type,
            dataset_path=args.dataset,
            metadata_path=args.metadata,
            embedding_manager=embedding_manager,
            max_images=args.max_images
        )
    else:
        # Load existing embeddings - try different possible names
        embedding_loaded = False
        possible_paths = [
            f"embeddings/{args.dataset_type}_dataset",
            "embeddings/dataset",  # Fallback to generic dataset
            f"embeddings/{args.dataset_type}"
        ]
        
        for path in possible_paths:
            try:
                embedding_manager.load_index(path)
                print(f"✅ Loaded embeddings from: {path}")
                embedding_loaded = True
                break
            except (FileNotFoundError, RuntimeError):
                continue
        
        if not embedding_loaded:
            print(f"❌ No existing embeddings found. Please run with --process-dataset first.")
            return
    
    # Initialize RAG and image generation components
    rag_manager = RAGManager(embedding_manager)
    image_generator = ImageGenerator()
    
    # Process the query if provided
    if args.prompt:
        print(f"Processing query: {args.prompt}")
        rag_output = rag_manager.process_query(args.prompt)
        
        print(f"Augmented prompt: {rag_output['augmented_prompt']}")
        print("\nRetrieved similar examples:")
        for example in rag_output['similar_examples']:
            print(f"- {example['metadata'].get('description', 'No description')}")
            print(f"  Tags: {', '.join(example['metadata'].get('tags', []))}")
            print()
        
        # Generate images
        print("Generating images...")
        result = image_generator.generate_from_rag_output(
            rag_output=rag_output,
            output_dir="output",
            num_images=args.num_images,
            seed=args.seed,
            fast_mode=args.fast_mode
        )
        
        print("\nGenerated images saved to:")
        for path in result['generated_images']:
            print(f"- {path}")

if __name__ == "__main__":
    main() 


    