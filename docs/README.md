# Image Model COCO - Documentation

## 📚 Documentation Index

### Getting Started
- [Installation Guide](installation.md)
- [Quick Start](quickstart.md)
- [Configuration](configuration.md)

### Core Features
- [Database Integration](database/README.md)
- [Embedding Management](embeddings/README.md)
- [Image Generation](generation/README.md)
- [RAG System](rag/README.md)

### Development
- [API Reference](api/README.md)
- [Architecture](architecture.md)
- [Contributing](contributing.md)

### Operations
- [Deployment](deployment.md)
- [Monitoring](monitoring.md)
- [Troubleshooting](troubleshooting.md)

## 🏗️ Project Architecture

```
image-model-coco-model/
├── src/                    # Core application code
│   ├── database/          # PostgreSQL database integration
│   ├── embedding/         # Embedding computation and management
│   ├── generation/        # Image generation pipeline
│   ├── retrieval/         # RAG retrieval system
│   ├── data_processing/   # COCO dataset processing
│   └── utils/             # Utility functions
├── examples/              # Usage examples and demos
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── config/                # Configuration files
├── docs/                  # Documentation
└── docker/                # Docker configuration
```

## 🚀 Quick Overview

This project provides:

- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Embedding Models**: Sentence Transformers and CLIP for text and image embeddings
- **RAG Pipeline**: Retrieval-augmented generation for context-aware image creation
- **COCO Integration**: Full COCO dataset support with metadata
- **Performance Monitoring**: Comprehensive system metrics and logging
- **Model Caching**: Intelligent model caching for improved performance

## 📖 Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Setup Database**: `make setup-db`
3. **Run Examples**: `python examples/basic_usage.py`

For detailed instructions, see the [Installation Guide](installation.md) and [Quick Start](quickstart.md).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 