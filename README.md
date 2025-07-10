# contextgen-ai

This project implements a Retrieval-Augmented Generation (RAG) system for AI image generation. The system uses a dataset of images and their associated metadata to guide the image generation process, ensuring that generated images maintain stylistic and conceptual consistency with the training data.

## Features

- Image and metadata processing pipeline
- Vector-based similarity search for relevant examples
- Integration with state-of-the-art image generation models
- RAG-based prompt enhancement
- Efficient storage and retrieval of image embeddings

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file with the following variable:
```
PEXELS_API_KEY=your_pexels_api_key_here
```

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Dataset storage (COCO, custom datasets)
├── docs/                   # Documentation
├── embeddings/             # Vector embeddings and FAISS indices
├── examples/               # Usage examples and tutorials
├── output/                 # Generated images
├── scripts/                # Utility scripts (dataset download, etc.)
├── src/                    # Core library code
│   ├── data_processing/    # Data loading and preprocessing
│   ├── embedding/          # Vector embedding generation
│   ├── generation/         # Image generation pipeline
│   ├── retrieval/          # RAG and similarity search
│   └── utils/             # Utility functions
└── tests/                  # Test files
```

## Quick Start

### 1. Download COCO Dataset
```bash
python scripts/download_coco.py
```

### 2. Process Dataset (Create Embeddings)
```bash
python -m src.main --process-dataset --dataset-type coco --max-images 1000
```

### 3. Generate Images
```bash
python -m src.main --prompt "a cat sitting on a chair" --num-images 2
```

### 4. Run Example
```bash
python examples/basic_usage.py
```

## Advanced Usage

### Configuration
Edit `config/config.py` to customize:
- Model settings (Stable Diffusion, CLIP, etc.)
- Embedding parameters
- Dataset options
- Device preferences

### Custom Datasets
To use your own dataset instead of COCO:
```bash
python -m src.main --process-dataset --dataset-type custom --dataset /path/to/images --metadata /path/to/metadata.csv
```

### API Usage
```python
from src.embedding.embedding_manager import EmbeddingManager
from src.retrieval.rag_manager import RAGManager
from src.generation.image_generator import ImageGenerator

# Initialize components
embedding_manager = EmbeddingManager()
embedding_manager.load_index("embeddings/coco_dataset")
rag_manager = RAGManager(embedding_manager)
image_generator = ImageGenerator()

# Generate images
rag_output = rag_manager.process_query("your prompt here")
result = image_generator.generate_from_rag_output(rag_output, "output/")
```


