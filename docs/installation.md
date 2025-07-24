# Installation Guide

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended)
- **Storage**: At least 10GB free space
- **Docker**: For database setup (optional)

### Required Software
- Python 3.8+
- pip (Python package manager)
- Git
- Docker & Docker Compose (for database)

## 🚀 Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-model-coco-model.git
cd image-model-coco-model
```

### 2. Create Virtual Environment

```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Setup Database (Optional but Recommended)

```bash
# Start PostgreSQL with Docker
make docker-up

# Setup database tables
make setup-db

# Verify connection
make test-db
```

### 5. Download Models (First Run)

The system will automatically download required models on first use:
- Sentence Transformers: `all-MiniLM-L6-v2`
- CLIP: `openai/clip-vit-base-patch32`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5433/image_rag_db

# Model Configuration
MODEL_CACHE_DIR=.model_cache
LOG_LEVEL=INFO

# Performance Configuration
MAX_WORKERS=4
BATCH_SIZE=32
```

### Configuration Files

- `config/config.py`: Main configuration settings
- `config/database.py`: Database-specific settings
- `config/models.py`: Model configuration

## 🧪 Verify Installation

### Run Basic Test

```bash
# Test basic functionality
python examples/basic_usage.py
```

### Test Database Integration

```bash
# Test database functionality
python examples/database_usage.py
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_database.py
```

## 🐳 Docker Installation (Alternative)

If you prefer using Docker for the entire application:

```bash
# Build the application
docker build -t image-model-coco .

# Run with database
docker-compose up -d
```

## 📦 Development Installation

For development work:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
make lint

# Run formatting
make format
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check if PostgreSQL is running
make docker-status

# Restart database
make docker-restart
```

#### 2. Model Download Issues
```bash
# Clear model cache
make clean-cache

# Check internet connection
curl -I https://huggingface.co
```

#### 3. Memory Issues
```bash
# Reduce batch size in config
# Set MODEL_CACHE_LIMIT_MB=1024
```

#### 4. Port Conflicts
```bash
# Check what's using port 5433
lsof -i :5433

# Change port in docker-compose.yml
```

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review [GitHub Issues](https://github.com/yourusername/image-model-coco-model/issues)
- Join our [Discord Community](https://discord.gg/your-community)

## 📚 Next Steps

After installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Explore [Examples](examples/)
3. Check the [API Reference](api/README.md)
4. Review [Configuration Options](configuration.md)

## 🔄 Updates

To update the project:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Update database schema (if needed)
make setup-db
``` 