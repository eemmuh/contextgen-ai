#!/usr/bin/env python3
"""
Advanced Model Caching Demo

This script demonstrates the enhanced model caching functionality:
1. First run: Models are downloaded and cached
2. Second run: Models are loaded from cache (much faster)
3. Advanced cache management features
4. Cache monitoring and optimization
5. Cache warmup capabilities
"""

import sys
import os
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedding_manager import EmbeddingManager
from src.generation.image_generator import ImageGenerator
from src.utils.model_cache import get_model_cache
from config.config import CACHE_CONFIG

def demo_basic_caching():
    """Demonstrate basic model caching functionality."""
    print("🎨 Basic Model Caching Demo")
    print("=" * 50)
    
    # Get cache instance
    model_cache = get_model_cache()
    
    # Show initial cache state
    print("\n1. Initial cache state:")
    cache_info = model_cache.get_cache_info()
    print(f"   Memory cache: {cache_info['memory_cache_size']} models")
    print(f"   Disk cache: {cache_info['disk_cache_size']} models")
    print(f"   Hit rate: {cache_info['hit_rate_percent']:.1f}%")
    
    # First run - models will be downloaded
    print("\n2. First run - Loading models (will download if not cached):")
    start_time = time.time()
    
    print("   Loading embedding models...")
    embedding_manager = EmbeddingManager()
    
    print("   Loading image generation model...")
    image_generator = ImageGenerator()
    
    first_run_time = time.time() - start_time
    print(f"   ⏱️ First run completed in {first_run_time:.2f} seconds")
    
    # Show cache state after first run
    cache_info = model_cache.get_cache_info()
    print(f"   📦 Models now cached: {cache_info['memory_cache_size']} in memory, {cache_info['disk_cache_size']} on disk")
    print(f"   💾 Total cache size: {cache_info['total_size_mb']:.1f} MB")
    
    # Second run - models should be loaded from cache
    print("\n3. Second run - Loading models from cache:")
    start_time = time.time()
    
    print("   Loading embedding models (from cache)...")
    embedding_manager2 = EmbeddingManager()
    
    print("   Loading image generation model (from cache)...")
    image_generator2 = ImageGenerator()
    
    second_run_time = time.time() - start_time
    print(f"   ⏱️ Second run completed in {second_run_time:.2f} seconds")
    
    # Show performance improvement
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"   🚀 Speedup: {speedup:.1f}x faster!")
    
    # Test cache functionality
    print("\n4. Testing cache functionality:")
    
    # Test text embedding
    print("   Testing text embedding...")
    text = "a beautiful sunset over mountains"
    embedding = embedding_manager.compute_text_embedding(text)
    print(f"   ✅ Text embedding computed: {embedding.shape}")
    
    # Test image generation (fast mode)
    print("   Testing image generation (fast mode)...")
    try:
        images = image_generator.generate_image(
            prompt="a simple test image",
            fast_mode=True,
            num_images=1
        )
        print(f"   ✅ Image generated: {len(images)} image(s)")
    except Exception as e:
        print(f"   ⚠️ Image generation failed: {e}")
    
    # Show final cache state
    print("\n5. Final cache state:")
    cache_info = model_cache.get_cache_info()
    print(f"   Memory cache: {cache_info['memory_cache_size']} models ({cache_info['memory_cache_size_mb']:.1f} MB)")
    print(f"   Disk cache: {cache_info['disk_cache_size']} models ({cache_info['disk_cache_size_mb']:.1f} MB)")
    print(f"   Total cache size: {cache_info['total_size_mb']:.1f} MB")
    print(f"   Hit rate: {cache_info['hit_rate_percent']:.1f}%")
    print(f"   Total requests: {cache_info['total_requests']}")
    print(f"   Cache hits: {cache_info['cache_hits']}")
    print(f"   Cache misses: {cache_info['cache_misses']}")

def demo_advanced_features():
    """Demonstrate advanced cache features."""
    print("\n🔧 Advanced Cache Features Demo")
    print("=" * 50)
    
    model_cache = get_model_cache()
    
    # 1. Cache warmup
    print("\n1. Cache Warmup:")
    print("   Warming up cache with frequently used models...")
    start_time = time.time()
    model_cache.warmup_cache(CACHE_CONFIG['warmup_models'])
    warmup_time = time.time() - start_time
    print(f"   ✅ Cache warmup completed in {warmup_time:.2f} seconds")
    
    # 2. Cache optimization
    print("\n2. Cache Optimization:")
    print("   Optimizing cache by removing old entries...")
    start_time = time.time()
    initial_info = model_cache.get_cache_info()
    initial_size = initial_info['total_size_mb']
    
    model_cache.optimize_cache()
    
    final_info = model_cache.get_cache_info()
    final_size = final_info['total_size_mb']
    optimization_time = time.time() - start_time
    space_saved = initial_size - final_size
    
    print(f"   ✅ Optimization completed in {optimization_time:.2f} seconds")
    print(f"   📉 Space saved: {space_saved:.1f} MB")
    
    # 3. Cache monitoring simulation
    print("\n3. Cache Monitoring Simulation:")
    print("   Simulating cache activity...")
    
    def simulate_cache_activity():
        """Simulate cache activity in background."""
        for i in range(10):
            time.sleep(0.5)
            # This will trigger cache hits
            try:
                embedding_manager = EmbeddingManager()
                embedding_manager.compute_text_embedding(f"test text {i}")
            except Exception:
                pass
    
    # Start background activity
    activity_thread = threading.Thread(target=simulate_cache_activity)
    activity_thread.start()
    
    # Monitor cache for 5 seconds
    print("   Monitoring cache performance for 5 seconds...")
    start_time = time.time()
    initial_stats = model_cache.get_cache_info()
    initial_requests = initial_stats['total_requests']
    
    print("   Time(s) | Memory(MB) | Disk(MB) | Hit Rate(%) | Requests/s")
    print("   " + "-" * 60)
    
    for i in range(5):
        time.sleep(1)
        info = model_cache.get_cache_info()
        elapsed = time.time() - start_time
        requests_diff = info['total_requests'] - initial_requests
        requests_per_sec = requests_diff / elapsed if elapsed > 0 else 0
        
        print(f"   {elapsed:>6.1f} | {info['memory_cache_size_mb']:>9.1f} | "
              f"{info['disk_cache_size_mb']:>7.1f} | {info['hit_rate_percent']:>10.1f} | "
              f"{requests_per_sec:>9.1f}")
    
    activity_thread.join()
    
    # 4. Cache statistics
    print("\n4. Cache Statistics:")
    info = model_cache.get_cache_info()
    print(f"   Total requests: {info['total_requests']}")
    print(f"   Cache hits: {info['cache_hits']}")
    print(f"   Cache misses: {info['cache_misses']}")
    print(f"   Hit rate: {info['hit_rate_percent']:.1f}%")
    print(f"   Evictions: {info['evictions']}")
    print(f"   Validation failures: {info['validation_failures']}")
    
    # 5. Cache limits and configuration
    print("\n5. Cache Configuration:")
    print(f"   Memory limit: {info['max_memory_size_mb']:.1f} MB")
    print(f"   Disk limit: {info['max_disk_size_mb']:.1f} MB")
    print(f"   Compression: {'Enabled' if info['compression_enabled'] else 'Disabled'}")

def demo_cache_management():
    """Demonstrate cache management features."""
    print("\n🔧 Cache Management Demo")
    print("=" * 50)
    
    model_cache = get_model_cache()
    
    # Show cache info
    print("1. Current cache information:")
    cache_info = model_cache.get_cache_info()
    print(f"   Cached models: {len(cache_info['cached_models'])}")
    
    # Clear specific model type
    if cache_info['cached_models']:
        model_type = list(cache_info['cached_models'].keys())[0]
        print(f"\n2. Clearing {model_type} models from cache...")
        model_cache.clear_cache(model_type)
        
        # Show updated cache info
        cache_info = model_cache.get_cache_info()
        print(f"   Remaining cached models: {len(cache_info['cached_models'])}")
    
    # Reset statistics
    print("\n3. Resetting cache statistics...")
    model_cache.reset_stats()
    print("   ✅ Statistics reset")
    
    # Show how to use cache manager script
    print("\n4. Cache Manager Script Usage:")
    print("   View cache info: python scripts/cache_manager.py info")
    print("   Clear all cache: python scripts/cache_manager.py clear")
    print("   Optimize cache: python scripts/cache_manager.py optimize")
    print("   Warm up cache: python scripts/cache_manager.py warmup")
    print("   Monitor cache: python scripts/cache_manager.py monitor")
    print("   Export cache info: python scripts/cache_manager.py export")

def demo_performance_comparison():
    """Demonstrate performance comparison with and without cache."""
    print("\n📊 Performance Comparison Demo")
    print("=" * 50)
    
    model_cache = get_model_cache()
    
    # Clear cache for fair comparison
    print("1. Clearing cache for fair comparison...")
    model_cache.clear_cache()
    
    # Test without cache
    print("\n2. Testing without cache (first run):")
    start_time = time.time()
    
    embedding_manager = EmbeddingManager()
    image_generator = ImageGenerator()
    
    no_cache_time = time.time() - start_time
    print(f"   ⏱️ Time without cache: {no_cache_time:.2f} seconds")
    
    # Test with cache (second run)
    print("\n3. Testing with cache (second run):")
    start_time = time.time()
    
    embedding_manager2 = EmbeddingManager()
    image_generator2 = ImageGenerator()
    
    with_cache_time = time.time() - start_time
    print(f"   ⏱️ Time with cache: {with_cache_time:.2f} seconds")
    
    # Calculate improvement
    if no_cache_time > 0:
        speedup = no_cache_time / with_cache_time
        time_saved = no_cache_time - with_cache_time
        improvement = ((no_cache_time - with_cache_time) / no_cache_time) * 100
        
        print(f"\n4. Performance Results:")
        print(f"   🚀 Speedup: {speedup:.1f}x faster")
        print(f"   ⏰ Time saved: {time_saved:.2f} seconds")
        print(f"   📈 Improvement: {improvement:.1f}%")
        
        # Show cache statistics
        cache_info = model_cache.get_cache_info()
        print(f"   📦 Models cached: {cache_info['memory_cache_size']} in memory, {cache_info['disk_cache_size']} on disk")
        print(f"   💾 Cache size: {cache_info['total_size_mb']:.1f} MB")
        print(f"   🎯 Hit rate: {cache_info['hit_rate_percent']:.1f}%")

if __name__ == "__main__":
    try:
        demo_basic_caching()
        demo_advanced_features()
        demo_cache_management()
        demo_performance_comparison()
        
        print("\n✅ Advanced model caching demo completed!")
        print("\n💡 Key Benefits of Enhanced Model Caching:")
        print("   - 🚀 Faster startup times with cache warmup")
        print("   - 📊 Comprehensive monitoring and statistics")
        print("   - 🔧 Automatic optimization and cleanup")
        print("   - 💾 Compression to save disk space")
        print("   - 🛡️ Model validation for reliability")
        print("   - 📈 LRU eviction for optimal memory usage")
        print("   - 🔒 Thread-safe operations")
        print("   - 📋 Detailed cache management tools")
        
        print("\n🛠️ Available Cache Management Commands:")
        print("   python scripts/cache_manager.py info      # View cache statistics")
        print("   python scripts/cache_manager.py monitor   # Real-time monitoring")
        print("   python scripts/cache_manager.py optimize  # Optimize cache")
        print("   python scripts/cache_manager.py warmup    # Warm up cache")
        print("   python scripts/cache_manager.py clear     # Clear cache")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 