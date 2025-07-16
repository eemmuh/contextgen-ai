"""
Configuration settings for the RAG-based Image Generation System.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
COCO_DATA_DIR = DATA_DIR / "coco"

# Model configurations
MODEL_CONFIG = {
    "stable_diffusion": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "num_inference_steps": 50,
        "guidance_scale": 7.5
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32"
    },
    "sentence_transformer": {
        "model_name": "all-MiniLM-L6-v2"
    }
}

# Cache configuration
CACHE_CONFIG = {
    "cache_dir": ".model_cache",
    "max_memory_size_mb": 2048,  # 2GB memory limit
    "max_disk_size_mb": 10240,   # 10GB disk limit
    "compression_enabled": True,
    "enable_validation": True,
    "warmup_models": [
        {
            "model_type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "device": "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"
        },
        {
            "model_type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "device": "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"
        }
    ],
    "optimization": {
        "auto_optimize": True,
        "optimization_interval_hours": 24,
        "max_age_days": 30
    }
}

# Embedding settings
EMBEDDING_CONFIG = {
    "embedding_dim": 384,
    "similarity_threshold": 0.5,
    "num_examples": 3
}

# Dataset settings
DATASET_CONFIG = {
    "max_images_dev": 1000,
    "max_images_prod": None,
    "default_split": "val2017"
}

# Device settings
DEVICE = "cuda" if os.getenv("FORCE_CPU") != "1" else "cpu"

# Environment variables
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, EMBEDDINGS_DIR, COCO_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 