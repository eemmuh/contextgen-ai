import os
import json
from typing import Dict, List, Optional
import torch
from PIL import Image
from transformers import CLIPProcessor
from .dataset import ImageMetadataDataset

class COCODataset(ImageMetadataDataset):
    def __init__(
        self,
        coco_dir: str,
        processor: Optional[CLIPProcessor] = None,
        max_images: Optional[int] = None,
        split: str = "train2017"
    ):
        """
        Initialize the COCO dataset adapter.
        
        Args:
            coco_dir: Directory containing COCO dataset
            processor: CLIP processor for image preprocessing
            max_images: Maximum number of images to load
            split: Dataset split to use (train2017 or val2017)
        """
        self.coco_dir = coco_dir
        self.split = split
        self.processor = processor or CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load processed annotations
        with open(os.path.join(coco_dir, "processed_annotations.json"), 'r') as f:
            self.annotations = json.load(f)
        
        # Filter for the specified split
        self.annotations = [
            ann for ann in self.annotations
            if ann['image_path'].startswith(os.path.join(coco_dir, split))
        ]
        
        if max_images:
            self.annotations = self.annotations[:max_images]
        
        # Validate image paths
        self.valid_images = []
        for idx, ann in enumerate(self.annotations):
            if os.path.exists(ann['image_path']):
                self.valid_images.append(idx)
        
        self.annotations = [self.annotations[i] for i in self.valid_images]
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - image: Processed image tensor
            - metadata: Dictionary of metadata fields
            - image_path: Path to the original image
        """
        ann = self.annotations[idx]
        
        # Load and process image
        image = Image.open(ann['image_path']).convert('RGB')
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        # Create metadata dictionary
        metadata = {
            'description': ' '.join(ann['captions']),
            'tags': ann['captions'],
            'width': ann['width'],
            'height': ann['height'],
            'id': ann['id']
        }
        
        return {
            'image': processed_image,
            'metadata': metadata,
            'image_path': ann['image_path']
        }
    
    def get_metadata_text(self, idx: int) -> str:
        """
        Get a formatted text representation of the metadata for a given index.
        """
        ann = self.annotations[idx]
        metadata_parts = []
        
        if ann['captions']:
            metadata_parts.append(f"Objects: {', '.join(ann['captions'])}")
        if ann.get('width') and ann.get('height'):
            metadata_parts.append(f"Size: {ann['width']}x{ann['height']}")
            
        return " | ".join(metadata_parts) 