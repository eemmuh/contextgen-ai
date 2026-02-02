# Database Integration

This document covers the PostgreSQL database integration with pgvector for vector similarity search and comprehensive data management.

## Overview

The system uses PostgreSQL with the pgvector extension to provide:
- **Vector similarity search** for fast image retrieval
- **Structured data storage** for images, embeddings, and metadata
- **ACID transactions** for data consistency
- **Scalable architecture** for large datasets
- **Performance monitoring** and analytics

## Database Schema

### Core Tables

#### Images Table
```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(500) UNIQUE NOT NULL,
    description TEXT,
    tags TEXT[],
    width INTEGER,
    height INTEGER,
    file_size_bytes INTEGER,
    format VARCHAR(10),
    source_dataset VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Embeddings Table (with pgvector)
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL,
    embedding vector(384),  -- Native pgvector for indexed similarity search
    model_type VARCHAR(100),
    model_name VARCHAR(200),
    embedding_type VARCHAR(50),  -- text, image, etc.
    embedding_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Why native `vector(384)` instead of JSON?**  
Storing embeddings in a native pgvector column is preferred over JSON because: (1) **performance** — the database can use IVFFlat or HNSW indexes for approximate nearest-neighbor search instead of scanning and scoring in application code; (2) **operators** — pgvector provides `<=>` (cosine distance), `<->` (L2), and index support; (3) **storage** — native vectors are compact and type-safe. The application uses the `pgvector` Python package and `Vector(384)` in SQLAlchemy when the extension is available; without it, a JSON fallback is used and similarity is computed in Python (not scalable for large tables).

#### Generations Table
```sql
CREATE TABLE generations (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    augmented_prompt TEXT,
    output_path VARCHAR(500),
    seed INTEGER,
    num_inference_steps INTEGER,
    guidance_scale FLOAT,
    generation_time_ms INTEGER,
    memory_usage_mb FLOAT,
    model_config JSONB,
    retrieved_examples JSONB,
    status VARCHAR(20) DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### System Metrics Table
```sql
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    cpu_usage FLOAT,
    memory_usage_mb FLOAT,
    disk_usage_mb FLOAT,
    gpu_memory_mb FLOAT,
    active_connections INTEGER,
    component VARCHAR(50)
);
```

#### Model Cache Table
```sql
CREATE TABLE model_cache (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    model_name VARCHAR(200),
    cache_path VARCHAR(500),
    size_mb FLOAT,
    last_accessed TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### User Sessions Table
```sql
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE,
    user_id VARCHAR(100),
    start_time TIMESTAMP DEFAULT NOW(),
    end_time TIMESTAMP,
    total_requests INTEGER DEFAULT 0,
    total_generations INTEGER DEFAULT 0,
    session_data JSONB
);
```

## Setup

### 1. Install PostgreSQL and pgvector

#### Ubuntu/Debian
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install pgvector
sudo apt install postgresql-15-pgvector  # Adjust version as needed
```

#### macOS
```bash
# Install PostgreSQL
brew install postgresql

# Install pgvector
brew install pgvector
```

#### Docker (Recommended)
```bash
# Use the provided docker-compose.yml
make docker-up
```

### 2. Create Database

```sql
-- Connect to PostgreSQL
sudo -u postgres psql

-- Create database and user
CREATE DATABASE image_rag_db;
CREATE USER image_rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE image_rag_db TO image_rag_user;

-- Connect to the database
\c image_rag_db

-- Enable pgvector extension
CREATE EXTENSION vector;
```

### 3. Setup Tables

```bash
# Automatic setup
make setup-db

# Manual setup
python scripts/setup_database.py
```

## Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql://postgres:password@localhost:5433/image_rag_db

# Connection pool settings
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_ECHO=false
```

### Configuration File

```python
# config/config.py
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5433/image_rag_db"),
    "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
    "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
    "echo": os.getenv("DB_ECHO", "false").lower() == "true"
}
```

## Usage Examples

### Database Manager

```python
from src.database.database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# Add image
image_id = db_manager.add_image(
    image_path="data/images/cat.jpg",
    description="A cute cat sitting on a chair",
    tags=["cat", "chair", "cute"],
    width=800,
    height=600
)

# Add embedding
db_manager.add_embedding(
    image_id=image_id,
    embedding=embedding_vector,
    model_type="sentence_transformer",
    model_name="all-MiniLM-L6-v2",
    embedding_type="text"
)

# Track generation
generation_id = db_manager.add_generation(
    prompt="a cat sitting on a chair",
    output_path="output/generated_cat.png",
    generation_time_ms=2500,
    memory_usage_mb=512.5
)

# Get statistics
stats = db_manager.get_database_stats()
print(f"Total images: {stats['total_images']}")
print(f"Total embeddings: {stats['total_embeddings']}")
```

### Embedding Manager

```python
from src.embedding.database_embedding_manager import DatabaseEmbeddingManager

# Initialize embedding manager
embedding_manager = DatabaseEmbeddingManager()

# Add image with embeddings
image_id = embedding_manager.add_image_with_embeddings(
    image_path="data/images/cat.jpg",
    metadata={
        "description": "A cute cat sitting on a chair",
        "tags": ["cat", "chair", "cute"],
        "width": 800,
        "height": 600
    }
)

# Search for similar images
results = embedding_manager.search_similar(
    query="a cat playing",
    k=5,
    similarity_threshold=0.5
)

for result in results:
    print(f"Image: {result['image_path']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
```

## Vector Similarity Search

### pgvector Operators

The system uses pgvector's similarity operators:

- `<->` : Cosine distance
- `<#>` : Negative inner product
- `<=>` : L2 distance

### Search Examples

```python
# Basic similarity search
results = embedding_manager.search_similar("a cat playing", k=5)

# Search with filters
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

### Performance Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_embeddings_model_type ON embeddings(model_type);
CREATE INDEX idx_embeddings_embedding_type ON embeddings(embedding_type);
CREATE INDEX idx_embeddings_created_at ON embeddings(created_at);

-- Create vector index for similarity search
CREATE INDEX idx_embeddings_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
```

## Monitoring and Analytics

### System Metrics

```python
# Add system metrics
db_manager.add_system_metrics(
    cpu_usage=45.2,
    memory_usage=2048.5,
    disk_usage=512.0,
    gpu_memory_mb=3072.0,
    active_connections=5,
    component="embedding"
)

# Get performance history
metrics = db_manager.get_system_metrics_history(hours=24)
```

### Database Statistics

```python
# Get comprehensive statistics
stats = db_manager.get_database_stats()

print(f"Total images: {stats['total_images']}")
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Total generations: {stats['total_generations']}")
print(f"Recent generations: {stats['recent_generations']}")
print(f"Average generation time: {stats['avg_generation_time_ms']:.1f}ms")
```

## Migration

### From FAISS to Database

```bash
# List available datasets
make list-datasets

# Dry run migration
make migrate-faiss

# Perform actual migration
make migrate-faiss-real

# Migrate specific dataset
python scripts/migrate_to_database.py --dataset coco_dataset --batch-size 50
```

### Migration Script

```python
from scripts.migrate_to_database import migrate_faiss_to_database

# Migrate FAISS index and metadata
migrate_faiss_to_database(
    faiss_index_path="embeddings/faiss_index.bin",
    metadata_path="embeddings/metadata.pkl",
    batch_size=100,
    dry_run=False
)
```

## Management Commands

### Makefile Commands

```bash
# Database setup
make setup-db          # Setup database tables
make verify-db         # Verify database setup
make test-db           # Test database connection

# Migration
make list-datasets     # List available FAISS datasets
make migrate-faiss     # Dry run migration
make migrate-faiss-real # Actual migration

# Docker management
make docker-up         # Start PostgreSQL container
make docker-down       # Stop PostgreSQL container
make docker-restart    # Restart PostgreSQL container
make docker-logs       # View database logs
make docker-status     # Check container status
```

### Direct Scripts

```bash
# Database setup
python scripts/setup_database.py

# Migration
python scripts/migrate_to_database.py --help

# Examples
python examples/database_usage.py
```

## Troubleshooting

### Common Issues

#### 1. Connection Failed
```bash
# Check if PostgreSQL is running
make docker-status

# Check connection
python -c "from src.database.session import test_connection; print('Connected!' if test_connection() else 'Failed')"
```

#### 2. pgvector Extension Missing
```sql
-- Connect to database and check extensions
\dx

-- Install pgvector if missing
CREATE EXTENSION vector;
```

#### 3. Port Conflicts
```bash
# Check what's using the port
lsof -i :5433

# Change port in docker-compose.yml
ports:
  - "5434:5432"  # Use different external port
```

#### 4. Memory Issues
```python
# Reduce connection pool size
DATABASE_CONFIG = {
    "pool_size": 5,
    "max_overflow": 10
}
```

### Performance Tuning

#### Connection Pooling
```python
# Optimize connection pool
DATABASE_CONFIG = {
    "pool_size": 20,        # Increase for high concurrency
    "max_overflow": 30,     # Allow more connections
    "pool_recycle": 3600,   # Recycle connections every hour
    "pool_pre_ping": True   # Validate connections before use
}
```

#### Query Optimization
```sql
-- Analyze table statistics
ANALYZE images;
ANALYZE embeddings;
ANALYZE generations;

-- Check query performance
EXPLAIN ANALYZE SELECT * FROM embeddings WHERE model_type = 'sentence_transformer';
```

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Database Setup Guide](../DATABASE_SETUP.md) 
