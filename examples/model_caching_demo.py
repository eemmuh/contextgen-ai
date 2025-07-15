#!/usr/bin/env python3
"""
Model Caching Demo

This script demonstrates the model caching functionality and its benefits:
1. First run: Models are downloaded and cached
2. Second run: Models are loaded from cache (much faster)
3. Cache management features
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding.embedding_manager import EmbeddingManager
from src.generation.image_generator import ImageGenerator
from src.utils.model_cache import get_model_cache

def demo_model_caching():
    """Demonstrate model caching functionality."""
    print("🎨 Model Caching Demo")
    print("=" * 50)
    
    # Get cache instance
    model_cache = get_model_cache()
    
    # Show initial cache state
    print("\n1. Initial cache state:")
    cache_info = model_cache.get_cache_info()
    print(f"   Memory cache: {cache_info['memory_cache_size']} models")
    print(f"   Disk cache: {cache_info['disk_cache_size']} models")
    
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
    print(f"   Memory cache: {cache_info['memory_cache_size']} models")
    print(f"   Disk cache: {cache_info['disk_cache_size']} models")
    print(f"   Total cache size: {cache_info['total_size_bytes'] / (1024*1024):.2f} MB")
    
    if cache_info['cached_models']:
        print("   Cached models:")
        for model_type, model_name in cache_info['cached_models'].items():
            print(f"     - {model_type}: {model_name}")

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
    
    # Show how to clear all cache
    print("\n3. To clear all cache, run:")
    print("   python -m src.main --clear-cache all")
    
    print("\n4. To view cache info, run:")
    print("   python -m src.main --cache-info")

if __name__ == "__main__":
    try:
        demo_model_caching()
        demo_cache_management()
        
        print("\n✅ Model caching demo completed!")
        print("\n💡 Benefits of model caching:")
        print("   - Faster startup times after first run")
        print("   - Reduced bandwidth usage")
        print("   - Offline capability for cached models")
        print("   - Memory efficiency with in-memory caching")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 