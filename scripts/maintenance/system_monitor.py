#!/usr/bin/env python3
"""
System Monitoring Script

This script provides comprehensive system monitoring for the RAG-based Image Generation System:
- Performance monitoring
- Health checks
- Cache management
- System diagnostics
- Real-time monitoring
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger_manager
from src.utils.performance_monitor import (
    get_performance_monitor,
    get_performance_summary,
    get_system_metrics,
)
from src.utils.health_check import (
    get_health_checker,
    run_health_check,
    get_health_summary,
)
from src.utils.model_cache import get_model_cache
from src.utils.error_handler import get_error_handler


def print_system_overview():
    """Print comprehensive system overview."""
    print("🔍 System Overview")
    print("=" * 60)

    # System metrics
    system_metrics = get_system_metrics()
    print(f"📊 System Metrics:")
    print(f"   Memory: {system_metrics['memory']['rss_mb']:.1f} MB ({system_metrics['memory']['percent']:.1f}%)")
    print(f"   CPU: {system_metrics['cpu']['percent']:.1f}% ({system_metrics['cpu']['count']} cores)")
    print(f"   Disk: {system_metrics['disk']['usage_percent']:.1f}% used")

    if system_metrics["gpu"]:
        print(f"   GPU: {len(system_metrics['gpu'])} device(s) available")
        for gpu_id, gpu_info in system_metrics["gpu"].items():
            print(f"     {gpu_id}: {gpu_info['memory_allocated_mb']:.1f} MB allocated")

    # Cache information
    cache = get_model_cache()
    cache_info = cache.get_cache_info()
    print(f"\n💾 Cache Information:")
    print(f"   Memory Cache: {cache_info['memory_cache_size']} models ({cache_info['memory_cache_size_mb']:.1f} MB)")
    print(f"   Disk Cache: {cache_info['disk_cache_size']} models ({cache_info['disk_cache_size_mb']:.1f} MB)")
    print(f"   Hit Rate: {cache_info['hit_rate_percent']:.1f}%")
    print(f"   Total Requests: {cache_info['total_requests']}")

    # Performance summary
    perf_summary = get_performance_summary()
    if perf_summary:
        print(f"\n⚡ Performance Summary:")
        print(f"   Total Operations: {perf_summary['total_operations']}")
        print(f"   Average Duration: {perf_summary['avg_duration']:.3f}s")
        print(f"   Max Duration: {perf_summary['max_duration']:.3f}s")
        print(f"   Average Memory Delta: {perf_summary['avg_memory_delta']:.1f} MB")

    # Health status
    health_summary = get_health_summary()
    print(f"\n🏥 Health Status:")
    print(f"   Overall Status: {health_summary['overall_status']}")
    print(f"   Healthy Checks: {health_summary['healthy_checks']}")
    print(f"   Warning Checks: {health_summary['warning_checks']}")
    print(f"   Critical Checks: {health_summary['critical_checks']}")


def print_detailed_health():
    """Print detailed health check results."""
    print("\n🏥 Detailed Health Check Results")
    print("=" * 60)

    results = run_health_check()

    for check_name, result in results.items():
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓",
        }

        emoji = status_emoji.get(result.status.value, "❓")
        print(f"{emoji} {check_name.upper()}: {result.message}")
        print(f"   Response Time: {result.response_time:.1f}ms")

        if result.details:
            for key, value in result.details.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value}")
                elif isinstance(value, dict):
                    print(f"   {key}: {json.dumps(value, indent=4)}")


def print_performance_details():
    """Print detailed performance information."""
    print("\n⚡ Detailed Performance Information")
    print("=" * 60)

    perf_summary = get_performance_summary()
    if not perf_summary:
        print("No performance data available")
        return

    print(f"Overall Performance:")
    print(f"   Total Operations: {perf_summary['total_operations']}")
    print(f"   Average Duration: {perf_summary['avg_duration']:.3f}s")
    print(f"   Max Duration: {perf_summary['max_duration']:.3f}s")
    print(f"   Min Duration: {perf_summary['min_duration']:.3f}s")
    print(f"   Average Memory Delta: {perf_summary['avg_memory_delta']:.1f} MB")
    print(f"   Max Memory Delta: {perf_summary['max_memory_delta']:.1f} MB")
    print(f"   Average CPU Usage: {perf_summary['avg_cpu_usage']:.1f}%")
    print(f"   Max CPU Usage: {perf_summary['max_cpu_usage']:.1f}%")

    if perf_summary["components"]:
        print(f"\nComponent Performance:")
        for component, stats in perf_summary["components"].items():
            print(f"   {component.upper()}:")
            print(f"     Operations: {stats['operation_count']}")
            print(f"     Avg Duration: {stats['avg_duration']:.3f}s")
            print(f"     Max Duration: {stats['max_duration']:.3f}s")
            print(f"     Avg Memory Delta: {stats['avg_memory_delta']:.1f} MB")


def print_cache_details():
    """Print detailed cache information."""
    print("\n💾 Detailed Cache Information")
    print("=" * 60)

    cache = get_model_cache()
    cache_info = cache.get_cache_info()

    print(f"Cache Statistics:")
    print(f"   Memory Cache Size: {cache_info['memory_cache_size']} models")
    print(f"   Memory Cache Size: {cache_info['memory_cache_size_mb']:.1f} MB")
    print(f"   Disk Cache Size: {cache_info['disk_cache_size']} models")
    print(f"   Disk Cache Size: {cache_info['disk_cache_size_mb']:.1f} MB")
    print(f"   Total Cache Size: {cache_info['total_size_mb']:.1f} MB")
    print(f"   Hit Rate: {cache_info['hit_rate_percent']:.1f}%")
    print(f"   Total Requests: {cache_info['total_requests']}")
    print(f"   Cache Hits: {cache_info['cache_hits']}")
    print(f"   Cache Misses: {cache_info['cache_misses']}")
    print(f"   Evictions: {cache_info['evictions']}")
    print(f"   Validation Failures: {cache_info['validation_failures']}")

    print(f"\nCache Limits:")
    print(f"   Memory Limit: {cache_info['max_memory_size_mb']:.1f} MB")
    print(f"   Disk Limit: {cache_info['max_disk_size_mb']:.1f} MB")
    print(f"   Compression: {'Enabled' if cache_info['compression_enabled'] else 'Disabled'}")

    if cache_info["cached_models"]:
        print(f"\nCached Models:")
        for model_type, models in cache_info["cached_models"].items():
            print(f"   {model_type}:")
            for model_name in models:
                print(f"     - {model_name}")


def real_time_monitoring(duration: int = 60, interval: int = 5):
    """Real-time monitoring with periodic updates."""
    print(f"\n📈 Real-time Monitoring (Duration: {duration}s, Interval: {interval}s)")
    print("=" * 60)
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)

    # Start background monitoring
    perf_monitor = get_performance_monitor()
    perf_monitor.start_monitoring(interval=2.0)

    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            current_time = datetime.now().strftime("%H:%M:%S")

            # Get current metrics
            system_metrics = get_system_metrics()
            cache_info = get_model_cache().get_cache_info()

            print(
                f"[{current_time}] "
                f"Memory: {system_metrics['memory']['rss_mb']:>6.1f}MB "
                f"CPU: {system_metrics['cpu']['percent']:>5.1f}% "
                f"Cache: {cache_info['memory_cache_size']:>2} models "
                f"Hit Rate: {cache_info['hit_rate_percent']:>5.1f}%"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    finally:
        perf_monitor.stop_monitoring()


def export_reports(output_dir: str = "reports"):
    """Export comprehensive system reports."""
    print(f"\n📄 Exporting Reports to {output_dir}")
    print("=" * 60)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export performance metrics
    perf_file = output_path / f"performance_report_{timestamp}.json"
    get_performance_monitor().export_metrics(str(perf_file))
    print(f"✅ Performance report: {perf_file}")

    # Export health report
    health_file = output_path / f"health_report_{timestamp}.json"
    get_health_checker().export_health_report(str(health_file))
    print(f"✅ Health report: {health_file}")

    # Export cache information
    cache_file = output_path / f"cache_report_{timestamp}.json"
    cache_info = get_model_cache().get_cache_info()
    with open(cache_file, "w") as f:
        json.dump(cache_info, f, indent=2, default=str)
    print(f"✅ Cache report: {cache_file}")

    # Export system metrics
    system_file = output_path / f"system_report_{timestamp}.json"
    system_metrics = get_system_metrics()
    with open(system_file, "w") as f:
        json.dump(system_metrics, f, indent=2, default=str)
    print(f"✅ System report: {system_file}")

    # Create summary report
    summary_file = output_path / f"summary_report_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "system_overview": system_metrics,
        "health_summary": get_health_summary(),
        "performance_summary": get_performance_summary(),
        "cache_summary": cache_info,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Summary report: {summary_file}")


def optimize_system():
    """Perform system optimization."""
    print("\n🔧 System Optimization")
    print("=" * 60)

    # Optimize memory
    print("Optimizing memory...")
    get_performance_monitor().optimize_memory()

    # Optimize cache
    print("Optimizing cache...")
    get_model_cache().optimize_cache()

    print("✅ System optimization completed")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="System Monitoring Tool")
    parser.add_argument(
        "command",
        choices=[
            "overview",
            "health",
            "performance",
            "cache",
            "monitor",
            "export",
            "optimize",
            "all",
        ],
        help="Command to execute",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=60,
        help="Monitoring duration in seconds (for monitor command)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=5,
        help="Monitoring interval in seconds (for monitor command)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="reports",
        help="Output directory for reports (for export command)",
    )

    args = parser.parse_args()

    try:
        if args.command == "overview":
            print_system_overview()
        elif args.command == "health":
            print_detailed_health()
        elif args.command == "performance":
            print_performance_details()
        elif args.command == "cache":
            print_cache_details()
        elif args.command == "monitor":
            real_time_monitoring(args.duration, args.interval)
        elif args.command == "export":
            export_reports(args.output_dir)
        elif args.command == "optimize":
            optimize_system()
        elif args.command == "all":
            print_system_overview()
            print_detailed_health()
            print_performance_details()
            print_cache_details()
            export_reports(args.output_dir)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
