#!/usr/bin/env python3
"""
Database analytics script for the RAG-based Image Generation System.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text

from src.database.database import DatabaseManager
from src.database.session import engine
from src.utils.logger import get_logger

logger = get_logger("database_analytics")


class DatabaseAnalytics:
    """Database analytics and reporting utility."""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.output_dir = Path("analytics")
        self.output_dir.mkdir(exist_ok=True)

    def get_basic_stats(self) -> dict:
        """Get basic database statistics."""
        try:
            with engine.connect() as conn:
                stats = {}

                # Table row counts
                tables = [
                    "images", "embeddings", "generations",
                    "system_metrics", "model_cache"
                ]
                for table in tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    stats[f"{table}_count"] = result.fetchone()[0]

                # Database size
                result = conn.execute(
                    text(
                        """
                    SELECT pg_size_pretty(pg_database_size(current_database())) 
                    as db_size
                """
                    )
                )
                stats["database_size"] = result.fetchone()[0]

                # Recent activity
                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM generations 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """
                    )
                )
                stats["generations_24h"] = result.fetchone()[0]

                result = conn.execute(
                    text(
                        """
                    SELECT COUNT(*) FROM generations 
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """
                    )
                )
                stats["generations_7d"] = result.fetchone()[0]

                # Success rate
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) 
                        as successful
                    FROM generations
                """
                    )
                )
                row = result.fetchone()
                if row[0] > 0:
                    stats["success_rate"] = (row[1] / row[0]) * 100
                else:
                    stats["success_rate"] = 0

                return stats

        except Exception as e:
            logger.error(f"Failed to get basic stats: {e}")
            return {}

    def get_generation_trends(self, days: int = 30) -> dict:
        """Get generation trends over time."""
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        f"""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as count,
                        AVG(generation_time_ms) as avg_time,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful
                    FROM generations
                    WHERE created_at > NOW() - INTERVAL '{days} days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """
                    )
                )

                trends = {
                    "dates": [], "counts": [], "avg_times": [], "success_rates": []
                }

                for row in result:
                    trends["dates"].append(row.date.strftime("%Y-%m-%d"))
                    trends["counts"].append(row.count)
                    trends["avg_times"].append(row.avg_time or 0)
                    success_rate = (
                        (row.successful / row.count * 100) if row.count > 0 else 0
                    )
                    trends["success_rates"].append(success_rate)

                return trends

        except Exception as e:
            logger.error(f"Failed to get generation trends: {e}")
            return {}

    def get_model_usage_stats(self) -> dict:
        """Get model usage statistics."""
        try:
            with engine.connect() as conn:
                # Embedding model usage
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        model_type,
                        model_name,
                        COUNT(*) as usage_count,
                        AVG(EXTRACT(EPOCH FROM (NOW() - created_at))/3600) as hours_ago
                    FROM embeddings
                    GROUP BY model_type, model_name
                    ORDER BY usage_count DESC
                """
                    )
                )

                embedding_stats = []
                for row in result:
                    embedding_stats.append(
                        {
                            "model_type": row.model_type,
                            "model_name": row.model_name,
                            "usage_count": row.usage_count,
                            "hours_ago": row.hours_ago,
                        }
                    )

                # Generation model usage
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        model_config->>'model_name' as model_name,
                        COUNT(*) as usage_count
                    FROM generations
                    WHERE model_config IS NOT NULL
                    GROUP BY model_config->>'model_name'
                    ORDER BY usage_count DESC
                """
                    )
                )

                generation_stats = []
                for row in result:
                    if row.model_name:
                        generation_stats.append({"model_name": row.model_name, "usage_count": row.usage_count})

                return {"embedding_models": embedding_stats, "generation_models": generation_stats}

        except Exception as e:
            logger.error(f"Failed to get model usage stats: {e}")
            return {}

    def get_performance_metrics(self) -> dict:
        """Get database performance metrics."""
        try:
            with engine.connect() as conn:
                # Connection stats
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

                # Table sizes
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """
                    )
                )

                table_sizes = []
                for row in result:
                    table_sizes.append({"table": row.tablename, "size": row.size})

                # Index usage
                result = conn.execute(
                    text(
                        """
                    SELECT 
                        indexname,
                        idx_scan as scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                    LIMIT 10
                """
                    )
                )

                index_stats = []
                for row in result:
                    index_stats.append(
                        {
                            "index": row.indexname,
                            "scans": row.scans,
                            "tuples_read": row.tuples_read,
                            "tuples_fetched": row.tuples_fetched,
                        }
                    )

                return {
                    "connections": {
                        "active": conn_stats[0] if conn_stats else 0,
                        "max": conn_stats[1] if conn_stats else 0,
                    },
                    "table_sizes": table_sizes,
                    "index_usage": index_stats,
                }

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}

    def generate_charts(self, trends: dict, output_path: str = None):
        """Generate charts from trends data."""
        if not trends or not trends.get("dates"):
            logger.warning("No trend data available for charts")
            return

        if output_path is None:
            output_path = self.output_dir / f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Database Analytics Dashboard", fontsize=16)

        # Generation count trend
        ax1.plot(trends["dates"], trends["counts"], marker="o", linewidth=2, markersize=6)
        ax1.set_title("Daily Generations")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # Average generation time
        ax2.plot(trends["dates"], trends["avg_times"], marker="s", color="orange", linewidth=2, markersize=6)
        ax2.set_title("Average Generation Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Time (ms)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Success rate
        ax3.plot(trends["dates"], trends["success_rates"], marker="^", color="green", linewidth=2, markersize=6)
        ax3.set_title("Success Rate")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Success Rate (%)")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)

        # Combined view
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(trends["dates"], trends["counts"], marker="o", color="blue", label="Count")
        line2 = ax4_twin.plot(trends["dates"], trends["avg_times"], marker="s", color="red", label="Time")
        ax4.set_title("Generations vs Time")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Count", color="blue")
        ax4_twin.set_ylabel("Time (ms)", color="red")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="upper left")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Charts saved to: {output_path}")

    def generate_report(self, output_format: str = "text", include_charts: bool = True) -> str:
        """Generate comprehensive analytics report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Collect all data
        basic_stats = self.get_basic_stats()
        trends = self.get_generation_trends()
        model_stats = self.get_model_usage_stats()
        performance = self.get_performance_metrics()

        # Generate charts if requested
        if include_charts and trends:
            chart_path = self.output_dir / f"charts_{timestamp}.png"
            self.generate_charts(trends, chart_path)

        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "basic_stats": basic_stats,
            "trends": trends,
            "model_stats": model_stats,
            "performance": performance,
        }

        if output_format == "json":
            output_path = self.output_dir / f"analytics_report_{timestamp}.json"
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
        else:
            output_path = self.output_dir / f"analytics_report_{timestamp}.txt"
            self._print_text_report(report, output_path)

        logger.info(f"Analytics report saved to: {output_path}")
        return str(output_path)

    def _print_text_report(self, report: dict, output_path: Path):
        """Print formatted text report."""
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("DATABASE ANALYTICS REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {report['timestamp']}\n\n")

            # Basic Stats
            f.write("BASIC STATISTICS\n")
            f.write("-" * 40 + "\n")
            stats = report["basic_stats"]
            f.write(f"Total Images: {stats.get('images_count', 0):,}\n")
            f.write(f"Total Embeddings: {stats.get('embeddings_count', 0):,}\n")
            f.write(f"Total Generations: {stats.get('generations_count', 0):,}\n")
            f.write(f"Database Size: {stats.get('database_size', 'Unknown')}\n")
            f.write(f"Generations (24h): {stats.get('generations_24h', 0):,}\n")
            f.write(f"Generations (7d): {stats.get('generations_7d', 0):,}\n")
            f.write(f"Success Rate: {stats.get('success_rate', 0):.1f}%\n\n")

            # Model Usage
            f.write("MODEL USAGE STATISTICS\n")
            f.write("-" * 40 + "\n")
            model_stats = report["model_stats"]

            f.write("Embedding Models:\n")
            for model in model_stats.get("embedding_models", [])[:5]:
                f.write(f"  {model['model_name']}: {model['usage_count']:,} uses\n")

            f.write("\nGeneration Models:\n")
            for model in model_stats.get("generation_models", [])[:5]:
                f.write(f"  {model['model_name']}: {model['usage_count']:,} uses\n")
            f.write("\n")

            # Performance
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            perf = report["performance"]
            f.write(f"Active Connections: {perf.get('connections', {}).get('active', 0)}\n")
            f.write(f"Max Connections: {perf.get('connections', {}).get('max', 0)}\n\n")

            f.write("Table Sizes:\n")
            for table in perf.get("table_sizes", [])[:5]:
                f.write(f"  {table['table']}: {table['size']}\n")
            f.write("\n")

            # Trends summary
            f.write("TRENDS SUMMARY (Last 30 Days)\n")
            f.write("-" * 40 + "\n")
            trends = report["trends"]
            if trends.get("counts"):
                f.write(f"Total Generations: {sum(trends['counts']):,}\n")
                f.write(f"Average Daily: {sum(trends['counts'])/len(trends['counts']):.1f}\n")
                f.write(f"Peak Day: {max(trends['counts']):,}\n")
                f.write(f"Average Generation Time: {sum(trends['avg_times'])/len(trends['avg_times']):.1f}ms\n")

            f.write("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Database analytics for RAG-based Image Generation System")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--trends-days", type=int, default=30, help="Number of days for trends (default: 30)")
    parser.add_argument("--output-dir", default="analytics", help="Output directory (default: analytics)")

    args = parser.parse_args()

    try:
        analytics = DatabaseAnalytics()

        # Generate report
        report_path = analytics.generate_report(output_format=args.format, include_charts=not args.no_charts)

        print(f"Analytics report generated: {report_path}")

        # Print basic stats to console
        stats = analytics.get_basic_stats()
        if stats:
            print("\nQuick Stats:")
            print(f"   Images: {stats.get('images_count', 0):,}")
            print(f"   Embeddings: {stats.get('embeddings_count', 0):,}")
            print(f"   Generations: {stats.get('generations_count', 0):,}")
            print(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")

        return 0

    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
