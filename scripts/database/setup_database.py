#!/usr/bin/env python3
"""
Database setup script for the RAG-based Image Generation System.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.session import create_tables, test_connection, engine
from src.database.models import Base
from src.utils.logger import get_logger
from sqlalchemy import text

logger = get_logger("database_setup")


def setup_pgvector_extension():
    """Setup pgvector extension in PostgreSQL."""
    try:
        with engine.connect() as conn:
            # Check if pgvector extension exists
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
            if not result.fetchone():
                logger.info("Installing pgvector extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension installed successfully")
            else:
                logger.info("pgvector extension already exists")
    except Exception as e:
        logger.error(f"Failed to setup pgvector extension: {e}")
        raise


def create_database_schema():
    """Create all database tables."""
    try:
        logger.info("Creating database tables...")
        create_tables()
        logger.info("Database schema created successfully")

        # Create performance indexes
        logger.info("Creating performance indexes...")
        with engine.connect() as conn:
            # Vector similarity search index
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """
                )
            )

            # Metadata indexes
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_embeddings_model_type 
                ON embeddings(model_type)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_type 
                ON embeddings(embedding_type)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_images_source_dataset 
                ON images(source_dataset)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_images_tags 
                ON images USING GIN(tags)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_generations_status 
                ON generations(status)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_generations_created_at 
                ON generations(created_at DESC)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp 
                ON system_metrics(timestamp DESC)
            """
                )
            )

            # Commit the changes
            conn.commit()

        logger.info("Performance indexes created successfully")

    except Exception as e:
        logger.error(f"Failed to create database schema: {e}")
        raise


def verify_database_setup():
    """Verify that the database is properly set up."""
    try:
        logger.info("Verifying database setup...")

        # Test connection
        if not test_connection():
            raise Exception("Database connection failed")

        # Check if tables exist
        with engine.connect() as conn:
            tables = ["images", "embeddings", "generations", "model_cache", "system_metrics", "user_sessions"]
            for table in tables:
                result = conn.execute(text(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table}'"))
                if not result.fetchone():
                    raise Exception(f"Table '{table}' not found")

        logger.info("Database setup verification completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database setup verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup database for RAG-based Image Generation System")
    parser.add_argument("--skip-extension", action="store_true", help="Skip pgvector extension setup")
    parser.add_argument("--verify-only", action="store_true", help="Only verify database setup")
    parser.add_argument("--drop-tables", action="store_true", help="Drop existing tables before creating new ones")

    args = parser.parse_args()

    try:
        if args.verify_only:
            if verify_database_setup():
                logger.info("Database setup verification passed")
                return 0
            else:
                logger.error("Database setup verification failed")
                return 1

        # Setup pgvector extension
        if not args.skip_extension:
            setup_pgvector_extension()

        # Drop tables if requested
        if args.drop_tables:
            from src.database.session import drop_tables

            logger.warning("Dropping existing tables...")
            drop_tables()

        # Create database schema
        create_database_schema()

        # Verify setup
        if verify_database_setup():
            logger.info("Database setup completed successfully")
            return 0
        else:
            logger.error("Database setup failed verification")
            return 1

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
