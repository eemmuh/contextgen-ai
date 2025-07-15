"""Tests for model caching functionality."""

import unittest
import sys
import os
import tempfile
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_cache import ModelCache, get_model_cache


class TestModelCache(unittest.TestCase):
    """Test model caching functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ModelCache(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        key1 = self.cache._get_cache_key("clip", "test-model", device="cpu")
        key2 = self.cache._get_cache_key("clip", "test-model", device="cpu")
        key3 = self.cache._get_cache_key("clip", "test-model", device="cuda")
        
        # Same parameters should generate same key
        self.assertEqual(key1, key2)
        
        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)
    
    def test_cache_info_empty(self):
        """Test cache info when cache is empty."""
        info = self.cache.get_cache_info()
        
        self.assertEqual(info['memory_cache_size'], 0)
        self.assertEqual(info['disk_cache_size'], 0)
        self.assertEqual(info['total_size_bytes'], 0)
        self.assertEqual(len(info['cached_models']), 0)
    
    def test_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_model_cache()
        cache2 = get_model_cache()
        
        self.assertIs(cache1, cache2)
    
    def test_cache_metadata_persistence(self):
        """Test that cache metadata is saved and loaded correctly."""
        # Add some test metadata
        test_metadata = {
            'test_key': {
                'model_type': 'test',
                'model_name': 'test-model',
                'device': 'cpu',
                'cache_path': '/test/path',
                'parameters': {},
                'created_at': 'test'
            }
        }
        
        self.cache.metadata = test_metadata
        self.cache._save_metadata()
        
        # Create new cache instance
        new_cache = ModelCache(cache_dir=self.temp_dir)
        
        # Check that metadata was loaded
        self.assertEqual(new_cache.metadata, test_metadata)


if __name__ == '__main__':
    unittest.main() 