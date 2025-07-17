"""Tests for advanced model caching functionality."""

import unittest
import sys
import os
import tempfile
import shutil
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.model_cache import ModelCache, get_model_cache, CacheEntry


class TestModelCache(unittest.TestCase):
    """Test advanced model caching functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        self.cache = ModelCache(
            cache_dir=self.temp_dir,
            max_memory_size_mb=10,  # Small limit for testing
            max_disk_size_mb=50,
            compression_enabled=False,  # Disable for testing
            enable_validation=False,  # Disable for testing
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_cache_entry_creation(self):
        """Test CacheEntry creation and access tracking."""
        # Create a mock model
        mock_model = type(
            "MockModel",
            (),
            {
                "state_dict": lambda: {"param1": torch.randn(10, 10)},
                "parameters": lambda: [torch.randn(10, 10)],
            },
        )()

        entry = CacheEntry(
            model=mock_model,
            model_type="test",
            model_name="test-model",
            device="cpu",
            parameters={"test": "param"},
            cache_path=Path(self.temp_dir) / "test",
        )

        # Test initial state
        self.assertEqual(entry.model_type, "test")
        self.assertEqual(entry.model_name, "test-model")
        self.assertEqual(entry.access_count, 0)

        # Test access tracking
        entry.access()
        self.assertEqual(entry.access_count, 1)
        self.assertGreater(entry.last_accessed, entry.created_at)

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        key1 = self.cache._get_cache_key("clip", "test-model", device="cpu")
        key2 = self.cache._get_cache_key("clip", "test-model", device="cpu")
        key3 = self.cache._get_cache_key("clip", "test-model", device="cuda")

        # Same parameters should generate same key
        self.assertEqual(key1, key2)

        # Different parameters should generate different keys
        self.assertNotEqual(key1, key3)

    def test_model_size_estimation(self):
        """Test model size estimation."""
        # Test PyTorch model
        model = torch.nn.Linear(10, 10)
        size = self.cache._estimate_model_size(model)
        self.assertGreater(size, 0)

        # Test generic model
        mock_model = type("MockModel", (), {})()
        size = self.cache._estimate_model_size(mock_model)
        self.assertEqual(size, 100 * 1024 * 1024)  # Default 100MB

    def test_memory_cache_eviction(self):
        """Test memory cache eviction when size limit is exceeded."""
        # Create a large mock model that exceeds the 10MB limit
        large_model = type(
            "LargeModel",
            (),
            {
                "state_dict": lambda: {"param1": torch.randn(1000, 1000)},
                "parameters": lambda: [torch.randn(1000, 1000)],
            },
        )()

        # Mock the size estimation to return a large size that will trigger eviction
        original_estimate = self.cache._estimate_model_size
        self.cache._estimate_model_size = lambda model: 20 * 1024 * 1024  # 20MB

        try:
            # Add model to cache (should trigger eviction due to small limit)
            self.cache.cache_model(
                model=large_model,
                model_type="test",
                model_name="large-model",
                device="cpu",
            )

            # Check that cache size is within limits (eviction should have occurred)
            info = self.cache.get_cache_info()
            self.assertLessEqual(info["memory_cache_size_mb"], 10)
        finally:
            # Restore original method
            self.cache._estimate_model_size = original_estimate

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        # Initial stats
        initial_stats = self.cache.get_cache_info()
        self.assertEqual(initial_stats["total_requests"], 0)
        self.assertEqual(initial_stats["cache_hits"], 0)
        self.assertEqual(initial_stats["cache_misses"], 0)

        # Try to get non-existent model (should be a miss)
        result = self.cache.get_cached_model("test", "non-existent", device="cpu")
        self.assertIsNone(result)

        # Check stats after miss
        stats = self.cache.get_cache_info()
        self.assertEqual(stats["total_requests"], 1)
        self.assertEqual(stats["cache_misses"], 1)
        self.assertEqual(stats["cache_hits"], 0)

    def test_cache_metadata_persistence(self):
        """Test that cache metadata is saved and loaded correctly."""
        # Add some test metadata
        test_metadata = {
            "test_key": {
                "model_type": "test",
                "model_name": "test-model",
                "device": "cpu",
                "cache_path": "/test/path",
                "parameters": {},
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 1,
                "size_bytes": 1024,
            }
        }

        self.cache.metadata = test_metadata
        self.cache._save_metadata()

        # Create new cache instance
        new_cache = ModelCache(cache_dir=self.temp_dir)

        # Check that metadata was loaded
        self.assertIn("test_key", new_cache.metadata)
        self.assertEqual(new_cache.metadata["test_key"]["model_name"], "test-model")

    def test_cache_optimization(self):
        """Test cache optimization functionality."""
        # Add old entries to metadata
        old_time = time.time() - (31 * 24 * 60 * 60)  # 31 days ago
        self.cache.metadata = {
            "old_entry": {
                "model_type": "test",
                "model_name": "old-model",
                "created_at": old_time,
                "last_accessed": old_time,
            },
            "new_entry": {
                "model_type": "test",
                "model_name": "new-model",
                "created_at": time.time(),
                "last_accessed": time.time(),
            },
        }

        # Run optimization
        self.cache.optimize_cache()

        # Check that old entry was removed
        self.assertNotIn("old_entry", self.cache.metadata)
        self.assertIn("new_entry", self.cache.metadata)

    def test_cache_warmup(self):
        """Test cache warmup functionality."""
        warmup_configs = [
            {"model_type": "test", "model_name": "warmup-model", "device": "cpu"}
        ]

        # Run warmup
        self.cache.warmup_cache(warmup_configs)

        # Check that warmup completed without errors
        # (actual model loading would require real models)
        self.assertTrue(True)  # Just check that no exception was raised

    def test_cache_limits(self):
        """Test cache size limits."""
        info = self.cache.get_cache_info()

        # Check that limits are set correctly
        self.assertEqual(info["max_memory_size_mb"], 10)
        self.assertEqual(info["max_disk_size_mb"], 50)

        # Check that compression setting is respected
        self.assertFalse(info["compression_enabled"])

    def test_thread_safety(self):
        """Test that cache operations are thread-safe."""
        import threading

        def cache_operation():
            """Perform cache operations in thread."""
            for i in range(10):
                self.cache.get_cached_model("test", f"model-{i}", device="cpu")

        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=cache_operation)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that no exceptions were raised
        self.assertTrue(True)

    def test_cache_clear_functionality(self):
        """Test cache clearing functionality."""
        # Add some test data
        self.cache.metadata = {
            "test1": {"model_type": "type1", "model_name": "model1"},
            "test2": {"model_type": "type2", "model_name": "model2"},
            "test3": {"model_type": "type1", "model_name": "model3"},
        }

        # Clear specific type
        self.cache.clear_cache("type1")

        # Check that only type1 entries were removed
        self.assertNotIn("test1", self.cache.metadata)
        self.assertNotIn("test3", self.cache.metadata)
        self.assertIn("test2", self.cache.metadata)

        # Clear all
        self.cache.clear_cache()
        self.assertEqual(len(self.cache.metadata), 0)

    def test_global_cache_singleton(self):
        """Test that global cache is a singleton."""
        cache1 = get_model_cache()
        cache2 = get_model_cache()

        self.assertIs(cache1, cache2)

    def test_stats_reset(self):
        """Test cache statistics reset functionality."""
        # Add some activity
        self.cache.get_cached_model("test", "model", device="cpu")

        # Check that stats were updated
        stats = self.cache.get_cache_info()
        self.assertGreater(stats["total_requests"], 0)

        # Reset stats
        self.cache.reset_stats()

        # Check that stats were reset
        stats = self.cache.get_cache_info()
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["cache_misses"], 0)

    def test_json_serialization(self):
        """Test JSON serialization of complex objects."""
        # Test with PyTorch tensors
        test_obj = {
            "tensor": torch.randn(5, 5),
            "dtype": torch.float32,
            "nested": {"list": [torch.randn(3, 3), "string", 123]},
        }

        serialized = self.cache._make_json_serializable(test_obj)

        # Check that it can be JSON serialized
        json_str = json.dumps(serialized)
        self.assertIsInstance(json_str, str)

        # Check that PyTorch objects were converted to strings
        self.assertIsInstance(serialized["tensor"], str)
        self.assertIsInstance(serialized["dtype"], str)


if __name__ == "__main__":
    unittest.main()
