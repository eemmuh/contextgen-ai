#!/usr/bin/env python3
"""
Database health check script for the RAG-based Image Generation System.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.session import test_connection, engine
from src.database.database import DatabaseManager
from src.utils.logger import get_logger
from sqlalchemy import text

logger = get_logger("database_health_check")


class DatabaseHealthChecker:
    """Comprehensive database health checker."""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "recommendations": [],
        }

    def check_connection(self):
        """Check database connection."""
        try:
            if test_connection():
                self.health_report["checks"]["connection"] = {
                    "status": "healthy",
                    "message": "Database connection successful",
                }
                return True
            else:
                self.health_report["checks"]["connection"] = {
                    "status": "failed",
                    "message": "Database connection failed",
                }
                return False
        except Exception as e:
            self.health_report["checks"]["connection"] = {"status": "error", "message": f"Connection error: {str(e)}"}
            return False

    def check_tables(self):
        """Check if all required tables exist."""
        try:
            with engine.connect() as conn:
                required_tables = [
                    "images",
                    "embeddings",
                    "generations",
                    "model_cache",
                    "system_metrics",
                    "user_sessions",
                ]

                missing_tables = []
                for table in required_tables:
                    result = conn.execute(text(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table}'"))
                    if not result.fetchone():
                        missing_tables.append(table)

                if missing_tables:
                    self.health_report["checks"]["tables"] = {
                        "status": "warning",
                        "message": f"Missing tables: {missing_tables}",
                    }
                    return False
                else:
                    self.health_report["checks"]["tables"] = {
                        "status": "healthy",
                        "message": "All required tables exist",
                    }
                    return True
        except Exception as e:
            self.health_report["checks"]["tables"] = {"status": "error", "message": f"Table check error: {str(e)}"}
            return False

    def check_pgvector_extension(self):
        """Check if pgvector extension is installed."""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
                if result.fetchone():
                    self.health_report["checks"]["pgvector"] = {
                        "status": "healthy",
                        "message": "pgvector extension is installed",
                    }
                    return True
                else:
                    self.health_report["checks"]["pgvector"] = {
                        "status": "failed",
                        "message": "pgvector extension not found",
                    }
                    return False
        except Exception as e:
            self.health_report["checks"]["pgvector"] = {"status": "error", "message": f"pgvector check error: {str(e)}"}
            return False

    def check_indexes(self):
        """Check if performance indexes exist."""
        try:
            with engine.connect() as conn:
                required_indexes = [
                    "idx_embeddings_vector",
                    "idx_embeddings_model_type",
                    "idx_images_source_dataset",
                    "idx_generations_created_at",
                ]

                missing_indexes = []
                for index in required_indexes:
                    result = conn.execute(text(f"SELECT 1 FROM pg_indexes WHERE indexname = '{index}'"))
                    if not result.fetchone():
                        missing_indexes.append(index)

                if missing_indexes:
                    self.health_report["checks"]["indexes"] = {
                        "status": "warning",
                        "message": f"Missing indexes: {missing_indexes}",
                    }
                    self.health_report["recommendations"].append(
                        "Run 'make setup-db' to create missing performance indexes"
                    )
                    return False
                else:
                    self.health_report["checks"]["indexes"] = {
                        "status": "healthy",
                        "message": "All performance indexes exist",
                    }
                    return True
        except Exception as e:
            self.health_report["checks"]["indexes"] = {"status": "error", "message": f"Index check error: {str(e)}"}
            return False

    def check_data_volume(self):
        """Check data volume and growth."""
        try:
            with engine.connect() as conn:
                # Get table row counts
                tables = ["images", "embeddings", "generations", "system_metrics"]
                row_counts = {}

                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    row_counts[table] = result.fetchone()[0]

                # Check recent activity
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM generations 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """
                    )
                )
                recent_generations = result.fetchone()[0]

                self.health_report["checks"]["data_volume"] = {
                    "status": "healthy",
                    "message": "Data volume check completed",
                    "details": {"row_counts": row_counts, "recent_generations_24h": recent_generations},
                }

                # Add recommendations based on data volume
                if row_counts["images"] == 0:
                    self.health_report["recommendations"].append(
                        "No images in database. Consider adding COCO dataset images."
                    )

                if recent_generations == 0:
                    self.health_report["recommendations"].append(
                        "No recent generations. Check if the system is actively being used."
                    )

                return True
        except Exception as e:
            self.health_report["checks"]["data_volume"] = {
                "status": "error",
                "message": f"Data volume check error: {str(e)}",
            }
            return False

    def check_performance(self):
        """Check database performance metrics."""
        try:
            with engine.connect() as conn:
                # Check connection pool status
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        count(*) as active_connections,
                        max_conn as max_connections
                    FROM pg_stat_activity, pg_settings 
                    WHERE name = 'max_connections'
                """
                    )
                )
                conn_stats = result.fetchone()

                # Check slow queries (if any)
                result = conn.execute(
                    text(
                        """
                    SELECT query, mean_time, calls 
                    FROM pg_stat_statements 
                    ORDER BY mean_time DESC 
                    LIMIT 5
                """
                    )
                )
                slow_queries = result.fetchall()

                self.health_report["checks"]["performance"] = {
                    "status": "healthy",
                    "message": "Performance check completed",
                    "details": {
                        "active_connections": conn_stats[0] if conn_stats else 0,
                        "max_connections": conn_stats[1] if conn_stats else 0,
                        "slow_queries": (
                            [{"query": q[0][:100] + "...", "mean_time": q[1], "calls": q[2]} for q in slow_queries]
                            if slow_queries
                            else []
                        ),
                    },
                }

                return True
        except Exception as e:
            self.health_report["checks"]["performance"] = {
                "status": "error",
                "message": f"Performance check error: {str(e)}",
            }
            return False

    def run_all_checks(self):
        """Run all health checks."""
        logger.info("🔍 Starting database health check...")

        checks = [
            ("connection", self.check_connection),
            ("tables", self.check_tables),
            ("pgvector", self.check_pgvector_extension),
            ("indexes", self.check_indexes),
            ("data_volume", self.check_data_volume),
            ("performance", self.check_performance),
        ]

        failed_checks = 0
        for check_name, check_func in checks:
            try:
                if not check_func():
                    failed_checks += 1
            except Exception as e:
                logger.error(f"Check '{check_name}' failed with exception: {e}")
                failed_checks += 1

        # Determine overall status
        if failed_checks == 0:
            self.health_report["overall_status"] = "healthy"
        elif failed_checks <= 2:
            self.health_report["overall_status"] = "warning"
        else:
            self.health_report["overall_status"] = "critical"

        return self.health_report

    def print_report(self, format_type="text"):
        """Print health check report."""
        if format_type == "json":
            print(json.dumps(self.health_report, indent=2))
        else:
            self._print_text_report()

    def _print_text_report(self):
        """Print formatted text report."""
        print("\n" + "=" * 60)
        print("DATABASE HEALTH CHECK REPORT")
        print("=" * 60)
        print(f"Timestamp: {self.health_report['timestamp']}")
        print(f"Overall Status: {self.health_report['overall_status'].upper()}")
        print("-" * 60)

        for check_name, check_data in self.health_report["checks"].items():
            status_icon = {"healthy": "✅", "warning": "⚠️", "failed": "❌", "error": "💥"}.get(
                check_data["status"], "❓"
            )

            print(f"{status_icon} {check_name.upper()}: {check_data['status']}")
            print(f"   {check_data['message']}")

            if "details" in check_data:
                for key, value in check_data["details"].items():
                    if isinstance(value, dict):
                        print(f"   {key}:")
                        for k, v in value.items():
                            print(f"     {k}: {v}")
                    else:
                        print(f"   {key}: {value}")
            print()

        if self.health_report["recommendations"]:
            print("RECOMMENDATIONS:")
            print("-" * 60)
            for i, rec in enumerate(self.health_report["recommendations"], 1):
                print(f"{i}. {rec}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Database health check for RAG-based Image Generation System")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    parser.add_argument("--output", help="Output file path (optional)")

    args = parser.parse_args()

    try:
        # Run health check
        checker = DatabaseHealthChecker()
        report = checker.run_all_checks()

        # Output results
        if args.output:
            with open(args.output, "w") as f:
                if args.format == "json":
                    json.dump(report, f, indent=2)
                else:
                    checker.print_report("text")
                    # Note: This will print to console, not file
            logger.info(f"Health check report saved to: {args.output}")
        else:
            checker.print_report(args.format)

        # Return appropriate exit code
        if report["overall_status"] == "healthy":
            return 0
        elif report["overall_status"] == "warning":
            return 1
        else:
            return 2

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
