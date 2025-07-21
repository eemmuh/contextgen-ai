#!/usr/bin/env python3
"""
Migration script to move from FAISS-based storage to PostgreSQL database.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embedding.database_embedding_manager import DatabaseEmbeddingManager
from src.database.session import test_connection
from src.utils.logger import get_logger

logger = get_logger("migration")


def find_faiss_files(embeddings_dir: str = "embeddings") -> list:
    """Find existing FAISS index and metadata files."""
    embeddings_path = Path(embeddings_dir)
    faiss_files = []
    
    if not embeddings_path.exists():
        logger.warning(f"Embeddings directory {embeddings_dir} not found")
        return faiss_files
    
    # Look for .index and .metadata files
    for file_path in embeddings_path.rglob("*.index"):
        metadata_path = file_path.with_suffix(".metadata")
        if metadata_path.exists():
            faiss_files.append({
                'index_path': str(file_path),
                'metadata_path': str(metadata_path),
                'name': file_path.stem
            })
    
    return faiss_files


def migrate_dataset(
    faiss_index_path: str,
    metadata_path: str,
    dataset_name: str,
    batch_size: int = 100,
    dry_run: bool = False
) -> int:
    """
    Migrate a single dataset from FAISS to database.
    
    Args:
        faiss_index_path: Path to FAISS index file
        metadata_path: Path to metadata pickle file
        dataset_name: Name of the dataset
        batch_size: Number of items to process in each batch
        dry_run: If True, only simulate the migration
    
    Returns:
        Number of items migrated
    """
    logger.info(f"Starting migration for dataset: {dataset_name}")
    
    if dry_run:
        logger.info("DRY RUN MODE - No actual migration will be performed")
    
    # Initialize database embedding manager
    embedding_manager = DatabaseEmbeddingManager()
    
    if not dry_run:
        # Perform actual migration
        migrated_count = embedding_manager.migrate_from_faiss(
            faiss_index_path=faiss_index_path,
            metadata_path=metadata_path,
            batch_size=batch_size
        )
    else:
        # Simulate migration
        import pickle
        with open(metadata_path, 'rb') as f:
            metadata_store = pickle.load(f)
        migrated_count = len(metadata_store)
        logger.info(f"Would migrate {migrated_count} items")
    
    logger.info(f"Migration completed for {dataset_name}: {migrated_count} items")
    return migrated_count


def main():
    parser = argparse.ArgumentParser(description="Migrate from FAISS to PostgreSQL database")
    parser.add_argument("--embeddings-dir", default="embeddings", help="Directory containing FAISS files")
    parser.add_argument("--dataset", help="Specific dataset to migrate (optional)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for migration")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without actually doing it")
    parser.add_argument("--list-only", action="store_true", help="Only list available datasets")
    
    args = parser.parse_args()
    
    try:
        # Check database connection
        if not args.dry_run and not args.list_only:
            if not test_connection():
                logger.error("Database connection failed. Please ensure PostgreSQL is running and configured.")
                return 1
        
        # Find FAISS files
        faiss_files = find_faiss_files(args.embeddings_dir)
        
        if not faiss_files:
            logger.warning("No FAISS files found for migration")
            return 0
        
        logger.info(f"Found {len(faiss_files)} FAISS datasets:")
        for i, file_info in enumerate(faiss_files):
            logger.info(f"  {i+1}. {file_info['name']}")
        
        if args.list_only:
            return 0
        
        # Filter by specific dataset if requested
        if args.dataset:
            faiss_files = [f for f in faiss_files if f['name'] == args.dataset]
            if not faiss_files:
                logger.error(f"Dataset '{args.dataset}' not found")
                return 1
        
        # Perform migration
        total_migrated = 0
        for file_info in faiss_files:
            try:
                migrated_count = migrate_dataset(
                    faiss_index_path=file_info['index_path'],
                    metadata_path=file_info['metadata_path'],
                    dataset_name=file_info['name'],
                    batch_size=args.batch_size,
                    dry_run=args.dry_run
                )
                total_migrated += migrated_count
            except Exception as e:
                logger.error(f"Failed to migrate dataset {file_info['name']}: {e}")
                continue
        
        if args.dry_run:
            logger.info(f"DRY RUN: Would migrate {total_migrated} total items")
        else:
            logger.info(f"Migration completed: {total_migrated} total items migrated")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 