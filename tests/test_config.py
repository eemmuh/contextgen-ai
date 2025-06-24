"""Tests for configuration module."""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_CONFIG, EMBEDDING_CONFIG, DATASET_CONFIG


class TestConfig(unittest.TestCase):
    """Test configuration settings."""
    
    def test_model_config_exists(self):
        """Test that model configuration is properly defined."""
        self.assertIn('stable_diffusion', MODEL_CONFIG)
        self.assertIn('clip', MODEL_CONFIG)
        self.assertIn('sentence_transformer', MODEL_CONFIG)
    
    def test_embedding_config_exists(self):
        """Test that embedding configuration is properly defined."""
        self.assertIn('embedding_dim', EMBEDDING_CONFIG)
        self.assertIn('similarity_threshold', EMBEDDING_CONFIG)
        self.assertIn('num_examples', EMBEDDING_CONFIG)
    
    def test_dataset_config_exists(self):
        """Test that dataset configuration is properly defined."""
        self.assertIn('max_images_dev', DATASET_CONFIG)
        self.assertIn('default_split', DATASET_CONFIG)
    
    def test_model_config_values(self):
        """Test that model configuration has reasonable values."""
        sd_config = MODEL_CONFIG['stable_diffusion']
        self.assertEqual(sd_config['num_inference_steps'], 50)
        self.assertEqual(sd_config['guidance_scale'], 7.5)
        self.assertIsInstance(sd_config['model_id'], str)


if __name__ == '__main__':
    unittest.main() 