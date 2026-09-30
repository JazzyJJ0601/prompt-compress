"""Tests for AdaptiveCompressor."""

import numpy as np
import sys
import os

# Add src directory to path for imports
project_root = os.path.join(os.path.dirname(__file__), "..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from prompt_compress.core import AdaptiveCompressor


def test_basic_compression():
    """Test basic compression functionality."""
    compressor = AdaptiveCompressor()
    
    tokens = ["hello", "world", "this", "is", "a", "test"]
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.5)
    
    assert len(drop_mask) == 6
    assert sum(drop_mask) >= 1  # At least one dropped


def test_segment_aware_allocation():
    """Test that budget is allocated across segments, not globally."""
    compressor = AdaptiveCompressor()
    
    # Create two segments with different importance levels
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])
    
    # With global budget, we'd keep top tokens from first segment
    # With segment allocation, each segment gets proportional budget
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.5)
    
    # Check that budget was applied (half tokens should be kept)
    kept_count = sum(1 for d in drop_mask if not d)
    assert 3 <= kept_count <= 5  # Allow some variance


def test_top_k_strategy():
    """Test top-k compression strategy."""
    compressor = AdaptiveCompressor(strategy="top-k")
    
    tokens = ["a", "b", "c", "d", "e"]
    scores = np.array([0.5, 0.9, 0.3, 0.8, 0.1])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.6)
    
    # Should keep highest importance tokens
    kept_count = sum(1 for d in drop_mask if not d)
    assert kept_count == 3  # 60% of 5


def test_threshold_strategy():
    """Test threshold-based compression strategy."""
    compressor = AdaptiveCompressor(strategy="threshold")
    
    tokens = ["a", "b", "c", "d", "e"]
    scores = np.array([0.9, 0.3, 0.7, 0.2, 0.8])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.6)
    
    kept_count = sum(1 for d in drop_mask if not d)
    assert kept_count > 0


def test_progressive_strategy():
    """Test progressive compression strategy."""
    compressor = AdaptiveCompressor(strategy="progressive")
    
    tokens = ["a", "b", "c", "d", "e", "f"]
    scores = np.array([0.5, 0.9, 0.3, 0.8, 0.1, 0.7])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.5)
    
    kept_count = sum(1 for d in drop_mask if not d)
    assert kept_count == 3  # 50% of 6


def test_placeholder_insertion():
    """Test that dropped tokens are replaced with placeholders."""
    compressor = AdaptiveCompressor(placeholder_token="[MASK]")
    
    tokens = ["hello", "world", "test"]
    scores = np.array([0.9, 0.1, 0.8])
    
    compressed, _ = compressor.compress(tokens, scores, budget=0.3)
    
    assert "[MASK]" in compressed


def test_reconstruction():
    """Test reconstruction preserves original structure."""
    compressor = AdaptiveCompressor()
    
    tokens = ["a", "b", "c", "d", "e"]
    scores = np.array([0.5, 0.9, 0.3, 0.8, 0.1])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=0.6)
    reconstructed, kept_indices = compressor.reconstruct(compressed, drop_mask, tokens)
    
    assert len(reconstructed) == len(tokens)
    assert len(kept_indices) > 0


def test_budget_as_int():
    """Test absolute token count budget."""
    compressor = AdaptiveCompressor()
    
    tokens = ["a", "b", "c", "d", "e", "f"]
    scores = np.array([0.5, 0.9, 0.3, 0.8, 0.1, 0.7])
    
    compressed, drop_mask = compressor.compress(tokens, scores, budget=3)
    
    kept_count = sum(1 for d in drop_mask if not d)
    assert kept_count == 3


def test_threshold_calculation():
    """Test dynamic threshold calculation."""
    compressor = AdaptiveCompressor()
    
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    threshold = compressor.get_threshold(scores, budget_ratio=0.5)
    
    assert 0.3 <= threshold <= 0.7
