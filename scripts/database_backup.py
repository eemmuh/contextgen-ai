#!/usr/bin/env python3
"""
Database backup script for the RAG-based Image Generation System.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json
import gzip
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.session import DATABASE_URL
from src.utils.logger import get_logger

logger = get_logger("database_backup")


class DatabaseBackup:
    """Database backup and restore utility."""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Parse database URL
        self.db_url = DATABASE_URL
        self.db_name = self._extract_db_name()
        self.db_host = self._extract_db_host()
        self.db_port = self._extract_db_port()
        self.db_user = self._extract_db_user()
        self.db_password = self._extract_db_password()
    
    def _extract_db_name(self) -> str:
        """Extract database name from URL."""
        return self.db_url.split('/')[-1].split('?')[0]
    
    def _extract_db_host(self) -> str:
        """Extract database host from URL."""
        return self.db_url.split('@')[1].split(':')[0]
    
    def _extract_db_port(self) -> str:
        """Extract database port from URL."""
        return self.db_url.split('@')[1].split(':')[1].split('/')[0]
    
    def _extract_db_user(self) -> str:
        """Extract database user from URL."""
        return self.db_url.split('://')[1].split(':')[0]
    
    def _extract_db_password(self) -> str:
        """Extract database password from URL."""
        return self.db_url.split('://')[1].split(':')[1].split('@')[0]
    
    def create_backup(self, compress: bool = True, include_metadata: bool = True) -> str:
        """Create a database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{self.db_name}_{timestamp}.sql"
        backup_path = self.backup_dir / backup_filename
        
        try:
            logger.info(f"Creating backup: {backup_path}")
            
            # Set environment variables for pg_dump
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            # Build pg_dump command
            cmd = [
                'pg_dump',
                f'--host={self.db_host}',
                f'--port={self.db_port}',
                f'--username={self.db_user}',
                '--verbose',
                '--clean',
                '--if-exists',
                '--no-owner',
                '--no-privileges'
            ]
            
            if include_metadata:
                cmd.extend(['--schema=public'])
            
            cmd.append(self.db_name)
            
            # Execute backup
            with open(backup_path, 'w') as f:
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Compress if requested
            if compress:
                compressed_path = backup_path.with_suffix('.sql.gz')
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                backup_path.unlink()
                backup_path = compressed_path
            
            # Create metadata file
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "database_name": self.db_name,
                "backup_filename": backup_path.name,
                "compressed": compress,
                "file_size_bytes": backup_path.stat().st_size,
                "pg_dump_version": self._get_pg_dump_version()
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Backup created successfully: {backup_path}")
            logger.info(f"   Size: {backup_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    def restore_backup(self, backup_path: str, drop_existing: bool = False) -> bool:
        """Restore database from backup."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        try:
            logger.info(f"Restoring backup: {backup_path}")
            
            # Set environment variables for psql
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_password
            
            # Determine if file is compressed
            is_compressed = backup_path.suffix == '.gz'
            
            # Build psql command
            cmd = [
                'psql',
                f'--host={self.db_host}',
                f'--port={self.db_port}',
                f'--username={self.db_user}',
                '--verbose',
                '--echo-all'
            ]
            
            if drop_existing:
                cmd.extend(['--clean', '--if-exists'])
            
            cmd.append(self.db_name)
            
            # Execute restore
            if is_compressed:
                with gzip.open(backup_path, 'rt') as f:
                    result = subprocess.run(
                        cmd,
                        env=env,
                        stdin=f,
                        stderr=subprocess.PIPE,
                        text=True
                    )
            else:
                with open(backup_path, 'r') as f:
                    result = subprocess.run(
                        cmd,
                        env=env,
                        stdin=f,
                        stderr=subprocess.PIPE,
                        text=True
                    )
            
            if result.returncode != 0:
                raise Exception(f"psql restore failed: {result.stderr}")
            
            logger.info("✅ Backup restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise
    
    def list_backups(self) -> list:
        """List available backups."""
        backups = []
        
        for file_path in self.backup_dir.glob("backup_*.sql*"):
            if file_path.suffix in ['.sql', '.gz']:
                # Find corresponding metadata file
                metadata_path = file_path.with_suffix('.json')
                metadata = {}
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                backup_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "metadata": metadata
                }
                backups.append(backup_info)
        
        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x["modified"], reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """Clean up old backups."""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        deleted_count = 0
        
        for file_path in self.backup_dir.glob("backup_*.sql*"):
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    
                    # Also delete metadata file if it exists
                    metadata_path = file_path.with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    logger.info(f"Deleted old backup: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        
        return deleted_count
    
    def _get_pg_dump_version(self) -> str:
        """Get pg_dump version."""
        try:
            result = subprocess.run(['pg_dump', '--version'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Database backup utility for RAG-based Image Generation System")
    parser.add_argument("action", choices=["backup", "restore", "list", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--backup-dir", default="backups",
                       help="Backup directory (default: backups)")
    parser.add_argument("--file", help="Backup file for restore action")
    parser.add_argument("--no-compress", action="store_true",
                       help="Don't compress backup files")
    parser.add_argument("--drop-existing", action="store_true",
                       help="Drop existing database before restore")
    parser.add_argument("--keep-days", type=int, default=30,
                       help="Keep backups for N days (default: 30)")
    
    args = parser.parse_args()
    
    try:
        backup_util = DatabaseBackup(args.backup_dir)
        
        if args.action == "backup":
            backup_path = backup_util.create_backup(
                compress=not args.no_compress,
                include_metadata=True
            )
            print(f"✅ Backup created: {backup_path}")
            
        elif args.action == "restore":
            if not args.file:
                print("❌ Error: --file argument required for restore action")
                return 1
            
            success = backup_util.restore_backup(args.file, args.drop_existing)
            if success:
                print("✅ Backup restored successfully")
            else:
                print("❌ Backup restore failed")
                return 1
                
        elif args.action == "list":
            backups = backup_util.list_backups()
            if not backups:
                print("No backups found")
            else:
                print(f"\nFound {len(backups)} backup(s):")
                print("-" * 80)
                for backup in backups:
                    size_mb = backup["size_bytes"] / 1024 / 1024
                    print(f"📁 {backup['filename']}")
                    print(f"   Size: {size_mb:.2f} MB")
                    print(f"   Modified: {backup['modified']}")
                    if backup["metadata"]:
                        print(f"   Database: {backup['metadata'].get('database_name', 'unknown')}")
                    print()
                    
        elif args.action == "cleanup":
            deleted_count = backup_util.cleanup_old_backups(args.keep_days)
            print(f"✅ Cleaned up {deleted_count} old backup(s)")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backup operation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 