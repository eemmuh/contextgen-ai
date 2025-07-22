# 🎨 Image Model COCO

**RAG-based Image Generation System with PostgreSQL Vector Database**

A comprehensive Retrieval-Augmented Generation (RAG) system for image generation using the COCO dataset, featuring PostgreSQL database integration with pgvector for efficient vector similarity search.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-✓-green.svg)](https://github.com/pgvector/pgvector)

## ✨ Features

- 🔍 **Vector Similarity Search**: Fast image retrieval using pgvector
- 🧠 **RAG Pipeline**: Context-aware image generation
- 🗄️ **PostgreSQL Database**: Scalable data storage with vector support
- 📊 **Performance Monitoring**: Comprehensive metrics and logging
- 🎯 **COCO Integration**: Full dataset support with metadata
- ⚡ **Model Caching**: Intelligent caching for improved performance
- 🐳 **Docker Support**: Easy deployment with Docker Compose

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/image-model-coco-model.git
cd image-model-coco-model

# Install dependencies
pip install -r requirements.txt

# Start database
make docker-up
make setup-db

# Run example
python examples/database_usage.py
```

## 📖 Documentation

- 📚 **[Full Documentation](docs/README.md)** - Complete project documentation
- ⚡ **[Quick Start](docs/quickstart.md)** - Get up and running in 5 minutes
- 🔧 **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- 🗄️ **[Database Setup](DATABASE_SETUP.md)** - PostgreSQL and pgvector configuration

## 🏗️ Architecture

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

## 🎯 Key Components

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

## 📊 Performance

- **Vector Search**: < 100ms for 1M+ embeddings
- **Image Processing**: Batch processing with configurable sizes
- **Database**: Connection pooling and optimized queries
- **Caching**: Intelligent model and result caching

## 🔧 Configuration

```python
# config/config.py
DATABASE_CONFIG = {
    "url": "postgresql://postgres:password@localhost:5433/image_rag_db",
    "pool_size": 10,
    "max_overflow": 20
}

MODEL_CONFIG = {
    "text_model": "all-MiniLM-L6-v2",
    "image_model": "openai/clip-vit-base-patch32",
    "cache_dir": ".model_cache"
}
```

## 📈 Usage Examples

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

## 🛠️ Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
make lint

# Run formatting
make format
```

### Project Structure
```
image-model-coco-model/
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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [COCO Dataset](https://cocodataset.org/) for the image dataset
- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector support
- [Sentence Transformers](https://www.sbert.net/) for text embeddings
- [CLIP](https://openai.com/research/clip) for image embeddings

## 📞 Support

- 📖 [Documentation](docs/README.md)
- 🐛 [Report Issues](https://github.com/yourusername/image-model-coco-model/issues)
- 💬 [Discord Community](https://discord.gg/your-community)
- 📧 [Email Support](mailto:support@yourproject.com)

---

**Made with ❤️ for the AI/ML community**


