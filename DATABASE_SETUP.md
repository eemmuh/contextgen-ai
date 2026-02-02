# Database Setup Guide

This guide will help you set up PostgreSQL with pgvector for the RAG-based Image Generation System.

## Quick Start (Recommended)

### Option 1: Docker (Easiest)

1. **Start PostgreSQL with Docker**
   ```bash
   make docker-up
   ```

2. **Verify the database is running**
   ```bash
   make docker-logs
   ```

3. **Setup database tables**
   ```bash
   make setup-db
   ```

4. **Test the connection**
   ```bash
   python -c "from src.database.session import test_connection; print('Connected' if test_connection() else 'Failed')"
   ```

5. **Run database examples**
   ```bash
   python examples/database_usage.py
   ```

### Option 2: Local PostgreSQL Installation

#### Prerequisites

- PostgreSQL 12+ installed
- pgvector extension installed
- Python dependencies installed (`uv pip install -e .`)

#### Installation Steps

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   ```

2. **Install pgvector Extension**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-14-pgvector
   
   # macOS
   brew install pgvector
   ```

3. **Create Database and User**
   ```bash
   # Connect to PostgreSQL
   sudo -u postgres psql
   
   # Create database and user
   CREATE DATABASE image_rag_db;
   CREATE USER image_rag_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE image_rag_db TO image_rag_user;
   
   # Connect to the database
   \c image_rag_db
   
   # Enable pgvector extension
   CREATE EXTENSION vector;
   
   # Exit
   \q
   ```

4. **Configure Environment**
   ```bash
   # Add to .env file
   echo "DATABASE_URL=postgresql://image_rag_user:your_password@localhost:5433/image_rag_db" >> .env
   ```

5. **Setup Database Tables**
   ```bash
   make setup-db
   ```

## Migration from FAISS

If you have existing FAISS-based embeddings, you can migrate them to the database:

1. **List available datasets**
   ```bash
   make list-datasets
   ```

2. **Dry run migration (simulate)**
   ```bash
   make migrate-faiss
   ```

3. **Perform actual migration**
   ```bash
   make migrate-faiss-real
   ```

4. **Migrate specific dataset**
   ```bash
   python scripts/migrate_to_database.py --dataset coco_dataset --batch-size 50
   ```

## Database Management

### Basic Commands

```bash
# Start/stop database
make docker-up          # Start PostgreSQL
make docker-down        # Stop PostgreSQL
make docker-logs        # View logs
make docker-reset       # Reset database (removes all data)

# Database operations
make setup-db           # Setup tables
make verify-db          # Verify setup
make list-datasets      # List FAISS datasets
make migrate-faiss      # Dry run migration
make migrate-faiss-real # Actual migration
```

### Connection Testing

```bash
# Test database connection
python -c "from src.database.session import test_connection; print('Connected!' if test_connection() else 'Failed')"

# Test with custom URL
DATABASE_URL=postgresql://user:pass@host:5432/db python -c "from src.database.session import test_connection; print('Connected!' if test_connection() else 'Failed')"
```

### Database Access

#### Using psql
```bash
   # Connect to database
   psql postgresql://postgres:password@localhost:5433/image_rag_db

# List tables
\dt

# View table structure
\d images
\d embeddings
\d generations

# Query examples
SELECT COUNT(*) FROM images;
SELECT COUNT(*) FROM embeddings;
SELECT prompt, generation_time_ms FROM generations ORDER BY created_at DESC LIMIT 5;
```

#### Using pgAdmin (Docker)
```bash
# Start pgAdmin
docker-compose up -d pgadmin

# Access pgAdmin at http://localhost:8080
# Login (default): admin@example.com / admin (configurable via docker-compose env vars)
 # Add server: postgres:5432 / postgres / password (use port 5433 for external connection)
```

## Configuration

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql://username:password@host:port/database

# Optional
DB_ECHO=false  # Set to true for SQL logging
```

### Configuration File

The database configuration is in `config/config.py`:

```python
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/image_rag_db"),
    "pool_size": 10,
    "max_overflow": 20,
    "pool_recycle": 3600,
    "echo": os.getenv("DB_ECHO", "false").lower() == "true"
}
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps
   # or
   sudo systemctl status postgresql
   ```

2. **pgvector Extension Not Found**
   ```bash
   # In PostgreSQL, check if extension is installed
   \dx
   
   # If not found, install it
   CREATE EXTENSION vector;
   ```

3. **Permission Denied**
   ```bash
   # Check user permissions
   GRANT ALL PRIVILEGES ON DATABASE image_rag_db TO your_user;
   GRANT ALL ON SCHEMA public TO your_user;
   ```

4. **Port Already in Use**
   ```bash
       # Check what's using port 5433
    sudo lsof -i :5433
   
   # Stop conflicting service or change port in docker-compose.yml
   ```

### Performance Tuning

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_images_source_dataset ON images(source_dataset);
CREATE INDEX CONCURRENTLY idx_embeddings_model_type ON embeddings(model_type);
CREATE INDEX CONCURRENTLY idx_generations_created_at ON generations(created_at);

-- Analyze tables for query optimization
ANALYZE images;
ANALYZE embeddings;
ANALYZE generations;
```

### Backup and Restore

```bash
   # Backup database
   pg_dump postgresql://postgres:password@localhost:5433/image_rag_db > backup.sql
   
   # Restore database
   psql postgresql://postgres:password@localhost:5433/image_rag_db < backup.sql
```

## Next Steps

After setting up the database:

1. **Run examples** to test the system
   ```bash
   python examples/database_usage.py
   ```

2. **Migrate existing data** if you have FAISS embeddings
   ```bash
   make migrate-faiss-real
   ```

3. **Update your code** to use the database-backed embedding manager
   ```python
   from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
   
   embedding_manager = DatabaseEmbeddingManager()
   ```

4. **Monitor performance** using the built-in analytics
   ```python
   stats = embedding_manager.get_database_stats()
   print(f"Total images: {stats['total_images']}")
   ```

## Support

If you encounter issues:

1. Check the logs: `make docker-logs`
2. Verify connection: `make verify-db`
3. Check the troubleshooting section above
4. Review the main README.md for more details 


