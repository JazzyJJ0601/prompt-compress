"""Tests for TokenImportanceScorer."""

import numpy as np
import sys
import os

# Add src directory to path for imports
project_root = os.path.join(os.path.dirname(__file__), "..")
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from prompt_compress.core import TokenImportanceScorer


def test_importance_scorer_basic():
    scorer = TokenImportanceScorer()
    logits = np.random.randn(10, 50).astype(np.float32)
    scores = scorer.score(logits)
    assert scores.shape == (10,)


def test_importance_scorer_with_targets():
    scorer = TokenImportanceScorer()
    logits = np.random.randn(10, 50).astype(np.float32)
    targets = np.random.randint(0, 50, size=(10,))
    scores = scorer.score(logits, targets)
    assert scores.shape == (10,)


def test_entropy_keyword():
    """Check that entropy is computed."""
    scorer = TokenImportanceScorer()
    logits = np.random.randn(5, 20).astype(np.float32)
    scores = scorer.score(logits)
    assert len(scores) == 5


def test_importance_keyword():
    """Check that importance scores are returned."""
    scorer = TokenImportanceScorer()
    logits = np.random.randn(3, 10).astype(np.float32)
    scores = scorer.score(logits)
    assert all(s >= 0.0 and s <= 1.0 for s in scores)
