# Image Model COCO Model

A RAG-based image generation system that uses COCO dataset for retrieval-augmented generation.

## Features

- **RAG-based Image Generation**: Uses COCO dataset for context-aware image generation
- **Advanced Model Caching**: Intelligent caching system with compression, validation, and monitoring
- **Multi-modal Embeddings**: CLIP and Sentence Transformers for text and image embeddings
- **FAISS Vector Search**: Fast similarity search for retrieving relevant examples
- **Stable Diffusion Integration**: High-quality image generation with context

## Enhanced Model Caching

The system now features an advanced model caching system with the following capabilities:

### 🚀 Key Features

- **Size Limits**: Configurable memory (2GB) and disk (10GB) limits with automatic eviction
- **LRU Eviction**: Least Recently Used policy for optimal cache management
- **Compression**: Automatic compression of cached models to save disk space
- **Model Validation**: Integrity checks for cached models to ensure reliability
- **Thread Safety**: Thread-safe operations for multi-threaded applications
- **Comprehensive Monitoring**: Real-time statistics and performance metrics
- **Cache Warmup**: Preloading of frequently used models
- **Automatic Optimization**: Background cleanup of old cache entries

### 📊 Cache Statistics

The cache provides detailed statistics including:
- Hit rate percentage
- Total requests, hits, and misses
- Memory and disk usage
- Eviction counts
- Validation failures
- Request rates

### 🛠️ Cache Management

Use the cache management script for various operations:

```bash
# View cache statistics
python scripts/cache_manager.py info

# Monitor cache performance in real-time
python scripts/cache_manager.py monitor

# Optimize cache (remove old entries)
python scripts/cache_manager.py optimize

# Warm up cache with frequently used models
python scripts/cache_manager.py warmup

# Clear specific model types
python scripts/cache_manager.py clear --model-type clip

# Clear all cache
python scripts/cache_manager.py clear

# Export cache information to JSON
python scripts/cache_manager.py export --output cache_report.json

# Reset cache statistics
python scripts/cache_manager.py reset-stats
```

### ⚙️ Configuration

Cache behavior can be configured in `config/config.py`:

```python
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
            "device": "cuda"
        },
        {
            "model_type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "device": "cuda"
        }
    ],
    "optimization": {
        "auto_optimize": True,
        "optimization_interval_hours": 24,
        "max_age_days": 30
    }
}
```

### 📈 Performance Benefits

- **Faster Startup**: Models load from cache instead of downloading
- **Reduced Bandwidth**: No repeated downloads of the same models
- **Memory Efficiency**: LRU eviction keeps most-used models in memory
- **Disk Space Optimization**: Compression and automatic cleanup
- **Reliability**: Model validation ensures cached models work correctly

### 🔧 Advanced Usage

#### Cache Warmup
```python
from src.utils.model_cache import get_model_cache
from config.config import CACHE_CONFIG

cache = get_model_cache()
cache.warmup_cache(CACHE_CONFIG['warmup_models'])
```

#### Custom Cache Configuration
```python
from src.utils.model_cache import ModelCache

# Create custom cache instance
custom_cache = ModelCache(
    cache_dir="custom_cache",
    max_memory_size_mb=4096,  # 4GB
    max_disk_size_mb=20480,   # 20GB
    compression_enabled=True,
    enable_validation=True
)
```

#### Cache Monitoring
```python
cache = get_model_cache()
info = cache.get_cache_info()

print(f"Hit Rate: {info['hit_rate_percent']:.1f}%")
print(f"Memory Usage: {info['memory_cache_size_mb']:.1f} MB")
print(f"Disk Usage: {info['disk_cache_size_mb']:.1f} MB")
print(f"Total Requests: {info['total_requests']}")
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image-model-coco-model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the COCO dataset:
```bash
python scripts/download_coco.py
```

## Usage

### Basic Usage

```python
from src.embedding.embedding_manager import EmbeddingManager
from src.generation.image_generator import ImageGenerator

# Initialize managers (models will be cached automatically)
embedding_manager = EmbeddingManager()
image_generator = ImageGenerator()

# Generate image with context
text = "a beautiful sunset over mountains"
embedding = embedding_manager.compute_text_embedding(text)
images = image_generator.generate_image(text, num_images=1)
```

### Demo Scripts

Run the enhanced caching demo:
```bash
python examples/model_caching_demo.py
```

Run basic usage demo:
```bash
python examples/basic_usage.py
```

## Project Structure

```
image-model-coco-model/
├── config/                 # Configuration files
├── data/                   # Dataset storage
├── embeddings/             # Pre-computed embeddings
├── examples/               # Demo scripts
├── output/                 # Generated images
├── scripts/                # Utility scripts
│   ├── cache_manager.py    # Cache management tool
│   └── download_coco.py    # Dataset downloader
├── src/                    # Source code
│   ├── data_processing/    # Dataset processing
│   ├── embedding/          # Embedding management
│   ├── generation/         # Image generation
│   ├── retrieval/          # RAG components
│   └── utils/              # Utilities including cache
└── tests/                  # Test files
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run specific cache tests:
```bash
python tests/test_model_cache.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


