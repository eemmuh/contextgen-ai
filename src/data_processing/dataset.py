import os
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from src.utils.model_cache import get_model_cache

class ImageMetadataDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        metadata_path: str,
        processor: Optional[CLIPProcessor] = None,
        max_images: Optional[int] = None
    ):
        """
        Initialize the dataset for image and metadata processing.
        
        Args:
            image_dir: Directory containing the images
            metadata_path: Path to the metadata CSV file
            processor: CLIP processor for image preprocessing
            max_images: Maximum number of images to load (for testing)
        """
        self.image_dir = image_dir
        self.processor = processor or self._load_clip_processor("openai/clip-vit-base-patch32")
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        if max_images:
            self.metadata = self.metadata.head(max_images)
            
        # Validate image paths
        self.valid_images = []
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(image_dir, row['image_filename'])
            if os.path.exists(img_path):
                self.valid_images.append(idx)
        
        self.metadata = self.metadata.iloc[self.valid_images].reset_index(drop=True)
    
    def _load_clip_processor(self, model_name: str) -> CLIPProcessor:
        """Load CLIP processor with caching."""
        # For processors, we'll use a simpler approach since they're lightweight
        # and don't need the full caching system
        return CLIPProcessor.from_pretrained(model_name)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - image: Processed image tensor
            - metadata: Dictionary of metadata fields
            - image_path: Path to the original image
        """
        row = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_filename'])
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        # Extract metadata
        metadata = {
            'description': row.get('description', ''),
            'tags': row.get('tags', '').split(',') if isinstance(row.get('tags'), str) else [],
            'style': row.get('style', ''),
            'category': row.get('category', '')
        }
        
        return {
            'image': processed_image,
            'metadata': metadata,
            'image_path': image_path
        }
    
    def get_metadata_text(self, idx: int) -> str:
        """
        Get a formatted text representation of the metadata for a given index.
        """
        row = self.metadata.iloc[idx]
        metadata_parts = []
        
        if row.get('description'):
            metadata_parts.append(f"Description: {row['description']}")
        if row.get('tags'):
            metadata_parts.append(f"Tags: {row['tags']}")
        if row.get('style'):
            metadata_parts.append(f"Style: {row['style']}")
        if row.get('category'):
            metadata_parts.append(f"Category: {row['category']}")
            
        return " | ".join(metadata_parts) 