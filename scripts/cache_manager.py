#!/usr/bin/env python3
"""
Cache Management Script

This script provides command-line tools for managing the model cache:
- View cache statistics
- Clear cache entries
- Optimize cache
- Warm up cache
- Monitor cache performance
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_cache import get_model_cache
from config.config import CACHE_CONFIG

def print_cache_info():
    """Print detailed cache information."""
    cache = get_model_cache()
    info = cache.get_cache_info()
    
    print("📊 Cache Information")
    print("=" * 50)
    print(f"Memory Cache: {info['memory_cache_size']} models ({info['memory_cache_size_mb']:.1f} MB)")
    print(f"Disk Cache: {info['disk_cache_size']} models ({info['disk_cache_size_mb']:.1f} MB)")
    print(f"Total Size: {info['total_size_mb']:.1f} MB")
    print(f"Hit Rate: {info['hit_rate_percent']:.1f}%")
    print(f"Total Requests: {info['total_requests']}")
    print(f"Cache Hits: {info['cache_hits']}")
    print(f"Cache Misses: {info['cache_misses']}")
    print(f"Evictions: {info['evictions']}")
    print(f"Validation Failures: {info['validation_failures']}")
    
    print(f"\nCache Limits:")
    print(f"  Memory Limit: {info['max_memory_size_mb']:.1f} MB")
    print(f"  Disk Limit: {info['max_disk_size_mb']:.1f} MB")
    print(f"  Compression: {'Enabled' if info['compression_enabled'] else 'Disabled'}")
    
    if info['cached_models']:
        print(f"\nCached Models:")
        for model_type, models in info['cached_models'].items():
            print(f"  {model_type}:")
            for model_name in models:
                print(f"    - {model_name}")

def clear_cache(model_type=None):
    """Clear cache entries."""
    cache = get_model_cache()
    
    if model_type:
        print(f"🗑️ Clearing {model_type} models from cache...")
        cache.clear_cache(model_type)
        print(f"✅ Cleared {model_type} models from cache")
    else:
        print("🗑️ Clearing all cache entries...")
        cache.clear_cache()
        print("✅ Cleared all cache entries")

def optimize_cache():
    """Optimize the cache by removing old entries."""
    cache = get_model_cache()
    
    print("🔧 Optimizing cache...")
    start_time = time.time()
    
    # Get initial stats
    initial_info = cache.get_cache_info()
    initial_size = initial_info['total_size_mb']
    
    # Run optimization
    cache.optimize_cache()
    
    # Get final stats
    final_info = cache.get_cache_info()
    final_size = final_info['total_size_mb']
    
    optimization_time = time.time() - start_time
    space_saved = initial_size - final_size
    
    print(f"✅ Cache optimization completed in {optimization_time:.2f} seconds")
    print(f"📉 Space saved: {space_saved:.1f} MB")
    print(f"📊 New total size: {final_size:.1f} MB")

def warmup_cache():
    """Warm up the cache with frequently used models."""
    cache = get_model_cache()
    
    print("🔥 Warming up cache...")
    start_time = time.time()
    
    warmup_models = CACHE_CONFIG.get('warmup_models', [])
    if not warmup_models:
        print("⚠️ No warmup models configured")
        return
    
    cache.warmup_cache(warmup_models)
    
    warmup_time = time.time() - start_time
    print(f"✅ Cache warmup completed in {warmup_time:.2f} seconds")

def reset_stats():
    """Reset cache statistics."""
    cache = get_model_cache()
    
    print("🔄 Resetting cache statistics...")
    cache.reset_stats()
    print("✅ Cache statistics reset")

def monitor_cache(duration=60, interval=5):
    """Monitor cache performance in real-time."""
    cache = get_model_cache()
    
    print(f"📈 Monitoring cache for {duration} seconds (interval: {interval}s)")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 80)
    print(f"{'Time':<8} {'Memory':<10} {'Disk':<10} {'Hit Rate':<10} {'Requests':<10}")
    print("-" * 80)
    
    start_time = time.time()
    initial_stats = cache.get_cache_info()
    initial_requests = initial_stats['total_requests']
    
    try:
        while time.time() - start_time < duration:
            info = cache.get_cache_info()
            
            # Calculate requests per second
            elapsed = time.time() - start_time
            requests_diff = info['total_requests'] - initial_requests
            requests_per_sec = requests_diff / elapsed if elapsed > 0 else 0
            
            print(f"{elapsed:>6.1f}s  {info['memory_cache_size_mb']:>8.1f}MB  "
                  f"{info['disk_cache_size_mb']:>8.1f}MB  {info['hit_rate_percent']:>8.1f}%  "
                  f"{requests_per_sec:>8.1f}/s")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")

def export_cache_info(output_file):
    """Export cache information to JSON file."""
    cache = get_model_cache()
    info = cache.get_cache_info()
    
    # Add timestamp
    info['exported_at'] = time.time()
    info['exported_at_iso'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📄 Cache information exported to {output_file}")

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Model Cache Management Tool")
    parser.add_argument('command', choices=[
        'info', 'clear', 'optimize', 'warmup', 'reset-stats', 
        'monitor', 'export'
    ], help='Command to execute')
    
    parser.add_argument('--model-type', '-t', 
                       help='Model type to clear (for clear command)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                       help='Monitoring duration in seconds (for monitor command)')
    parser.add_argument('--interval', '-i', type=int, default=5,
                       help='Monitoring interval in seconds (for monitor command)')
    parser.add_argument('--output', '-o', default='cache_info.json',
                       help='Output file for export command')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'info':
            print_cache_info()
        elif args.command == 'clear':
            clear_cache(args.model_type)
        elif args.command == 'optimize':
            optimize_cache()
        elif args.command == 'warmup':
            warmup_cache()
        elif args.command == 'reset-stats':
            reset_stats()
        elif args.command == 'monitor':
            monitor_cache(args.duration, args.interval)
        elif args.command == 'export':
            export_cache_info(args.output)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 