# contextgen-ai

**RAG-based Image Generation System with PostgreSQL Vector Database**

A comprehensive Retrieval-Augmented Generation (RAG) system for image generation using the COCO dataset, featuring PostgreSQL database integration with pgvector for efficient vector similarity search.

## Features

- **Vector Similarity Search**: Fast image retrieval using pgvector
- **RAG Pipeline**: Context-aware image generation
- **PostgreSQL Database**: Scalable data storage with vector support
- **Performance Monitoring**: Comprehensive metrics and logging
- **COCO Integration**: Full dataset support with metadata
- **Model Caching**: Intelligent caching for improved performance
- **Docker Support**: Easy deployment with Docker Compose

## Quick Start

```bash
# Clone and setup
git clone https://github.com/eemmuh/contextgen-ai.git
cd contextgen-ai

# Install dependencies
uv pip install -e .
# or (without uv):
# pip install -e .

# Start database
make docker-up
make setup-db

# Run example
python examples/database_usage.py
```

## Documentation

- **[Full Documentation](docs/README.md)** - Complete project documentation
- **[Quick Start](docs/quickstart.md)** - Get up and running in 5 minutes
- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Database Setup](DATABASE_SETUP.md)** - PostgreSQL and pgvector configuration

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   COCO Dataset  │    │  PostgreSQL DB  │    │  RAG Pipeline   │
│                 │    │   + pgvector    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Data Processing │    │ Vector Storage  │    │ Image Generation│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Components

### Database Integration
- **PostgreSQL** with **pgvector** extension
- **Vector similarity search** for fast image retrieval
- **Comprehensive metadata** storage and indexing
- **Performance monitoring** and analytics

### Embedding Management
- **Sentence Transformers** for text embeddings
- **CLIP** for image embeddings
- **Automatic model caching** and optimization
- **Batch processing** capabilities

### RAG System
- **Context-aware** image generation
- **Similarity-based** retrieval
- **Prompt augmentation** with retrieved examples
- **Generation tracking** and history

## Performance

- Batch-oriented dataset processing and embedding generation
- Connection pooling for database access
- Model and result caching

## Configuration

- App and model settings live in `config/settings.py` and `config/config.py`.
- Copy `env.example` to `.env` and adjust values as needed.

## Usage Examples

### Basic Image Search
```python
from src.embedding.database_embedding_manager import DatabaseEmbeddingManager

embedding_manager = DatabaseEmbeddingManager()
results = embedding_manager.search_similar("a cat playing", k=5)

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
```

### RAG-based Generation
```python
from src.generation.image_generator import ImageGenerator
from src.retrieval.rag_manager import RAGManager

rag_manager = RAGManager()
generator = ImageGenerator()

# Generate with context
prompt = "a cat playing with a ball"
augmented_prompt = rag_manager.augment_prompt(prompt)
generated_image = generator.generate(augmented_prompt)
```

## Development

### Setup Development Environment
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
make lint

# Run formatting
make format
```

### Project Structure
```
contextgen-ai/
├── src/                    # Core application code
│   ├── database/          # PostgreSQL integration
│   ├── embedding/         # Embedding management
│   ├── generation/        # Image generation
│   ├── retrieval/         # RAG system
│   ├── data_processing/   # COCO processing
│   └── utils/             # Utilities
├── examples/              # Usage examples
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── config/                # Configuration
├── docs/                  # Documentation
└── docker/                # Docker config
```


