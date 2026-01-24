# Quick Start Guide

## ⚡ Quick Setup

### 1. Install Dependencies

```bash
# Clone and setup
git clone https://github.com/eemmuh/contextgen-ai.git
cd contextgen-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
uv pip install -e .
# or (without uv):
# pip install -e .
```

### 2. Start Database

```bash
# Start PostgreSQL with Docker
make docker-up

# Setup database tables
make setup-db
```

### 3. Run Your First Example

```bash
# Basic usage example
python examples/basic_usage.py

# Database integration example
python examples/database_usage.py
```

## 🎯 Basic Usage

### Simple Image Search

```python
from src.embedding.database_embedding_manager import DatabaseEmbeddingManager

# Initialize embedding manager
embedding_manager = DatabaseEmbeddingManager()

# Search for similar images
results = embedding_manager.search_similar("a cat playing", k=5)

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
```

### Add Images to Database

```python
from src.database.database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# Add an image with metadata
image_id = db_manager.add_image(
    image_path="path/to/image.jpg",
    description="A cute cat sitting on a chair",
    tags=["cat", "pet", "indoor"],
    width=800,
    height=600
)

print(f"Added image with ID: {image_id}")
```

### Generate Images with RAG

```python
from src.generation.image_generator import ImageGenerator
from src.retrieval.rag_manager import RAGManager

# Initialize components
rag_manager = RAGManager()
generator = ImageGenerator()

# Generate image with RAG
prompt = "a cat playing with a ball"
augmented_prompt = rag_manager.augment_prompt(prompt)
generated_image = generator.generate(augmented_prompt)

# Save the result
generated_image.save("output/generated_cat.png")
```

## 📊 Database Operations

### View Database Statistics

```python
from src.database.database import DatabaseManager

db_manager = DatabaseManager()
stats = db_manager.get_database_stats()

print(f"Total images: {stats['total_images']}")
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Recent generations: {stats['recent_generations']}")
```

### Search with Filters

```python
# Search by model type
results = embedding_manager.search_similar(
    "outdoor scene",
    model_type="sentence_transformer",
    embedding_type="text",
    k=10
)

# Search with similarity threshold
results = embedding_manager.search_similar(
    "mountain landscape",
    similarity_threshold=0.5,
    k=5
)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (you can start from `env.example`):

```bash
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5433/image_rag_db

# Models
MODEL_CACHE_DIR=.model_cache
LOG_LEVEL=INFO

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
```

### Model Configuration

```python
# In config/config.py
MODEL_CONFIG = {
    "text_model": "all-MiniLM-L6-v2",
    "image_model": "openai/clip-vit-base-patch32",
    "cache_dir": ".model_cache",
    "device": "auto"  # or "cpu", "cuda"
}
```

## 🚀 Advanced Features

### Batch Processing

```python
# Process multiple images
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
metadata_list = [
    {"description": "A cat", "tags": ["cat"]},
    {"description": "A dog", "tags": ["dog"]},
    {"description": "A landscape", "tags": ["nature"]}
]

for path, metadata in zip(image_paths, metadata_list):
    embedding_manager.add_image_with_embeddings(path, metadata)
```

### Performance Monitoring

```python
from src.utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()

# Track generation performance
with monitor.track_generation():
    result = generator.generate("a beautiful sunset")
    
print(f"Generation time: {monitor.last_generation_time:.2f}s")
print(f"Memory usage: {monitor.last_memory_usage:.1f}MB")
```

### Model Caching

```python
from src.utils.model_cache import ModelCache

# Initialize cache
cache = ModelCache(
    memory_limit_mb=2048,
    disk_limit_mb=10240
)

# Models are automatically cached
embedding_manager = DatabaseEmbeddingManager()
# First load: downloads and caches
# Subsequent loads: uses cached version
```

## 📈 Monitoring & Logging

### View Logs

```bash
# View application logs
tail -f logs/app.log

# View database logs
make docker-logs
```

### System Metrics

```python
# Add system metrics
db_manager.add_system_metrics(
    cpu_usage=45.2,
    memory_usage=2048.5,
    disk_usage=512.0,
    active_connections=5
)

# Get performance history
metrics = db_manager.get_system_metrics_history(hours=24)
```

## 🎨 Example Workflows

### Complete RAG Pipeline

```python
def generate_with_rag(prompt: str, num_results: int = 3):
    """Complete RAG-based image generation pipeline."""
    
    # 1. Retrieve similar images
    similar_images = embedding_manager.search_similar(prompt, k=num_results)
    
    # 2. Augment prompt with retrieved context
    context = [img['description'] for img in similar_images]
    augmented_prompt = f"{prompt} (similar to: {', '.join(context)})"
    
    # 3. Generate image
    generated_image = generator.generate(augmented_prompt)
    
    # 4. Track generation
    db_manager.add_generation(
        prompt=prompt,
        augmented_prompt=augmented_prompt,
        retrieved_examples=similar_images,
        generation_time_ms=generator.last_generation_time * 1000
    )
    
    return generated_image, similar_images

# Usage
image, context = generate_with_rag("a cat playing in a garden")
```

## 🔍 Troubleshooting

### Common Issues

**Database Connection Failed:**
```bash
make docker-status
make docker-restart
```

**Model Download Issues:**
```bash
make clean-cache
python -c "from src.embedding.database_embedding_manager import DatabaseEmbeddingManager; DatabaseEmbeddingManager()"
```

**Memory Issues:**
```python
# Reduce batch size in config
MODEL_CONFIG["batch_size"] = 16
```

## 📚 Next Steps

1. **Explore Examples**: Check out all examples in `examples/`
2. **Read Documentation**: Visit the [full documentation](README.md)
3. **Customize**: Modify configuration for your use case
4. **Scale Up**: Add more images and embeddings
5. **Contribute**: Help improve the project!

## 🆘 Need Help?

- 📖 [Full Documentation](README.md)
- 🐛 Report issues via GitHub Issues (in this repository)