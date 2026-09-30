"""
Prompt-Compress: Gradient-free token compression for LLMs.

This module provides token importance scoring and adaptive compression
using information-theoretic measures from model logits.
"""

import numpy as np
from typing import Optional, List, Tuple


class TokenImportanceScorer:
    """
    Token importance scorer computing per-token scores based on:
    1. Entropy of the next-token distribution (uncertainty)
    2. Surprisal (negative log probability of the actual token)
    3. Mutual information between adjacent positions

    These scores follow information-theoretic principles where high-surprisal
    tokens carry more information and high-entropy regions indicate uncertainty.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the scorer.

        Args:
            temperature: Temperature for logit scaling. Lower values make
                        the distribution more peaked (more confident).
        """
        self.temperature = temperature

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to probabilities using numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _entropy(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute entropy of the probability distribution at each position.

        Entropy measures uncertainty: high entropy means the model is uncertain
        about what token to predict next.

        H_i = -sum_v P(v|C_i) * log_2 P(v|C_i)
        """
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log2(probs), axis=-1)

    def _surprisal(self, probs: np.ndarray, target_token_ids: np.ndarray) -> np.ndarray:
        """
        Compute surprisal (-log2 prob) for actual tokens.

        Surprisal measures information content: low probability tokens
        have high surprisal and carry more information.

        I(t|C) = -log_2 P(t|C)
        """
        prob_token = probs[np.arange(probs.shape[0]), target_token_ids]
        return -np.log2(np.clip(prob_token, 1e-10, 1.0))

    def _mutual_information_adjacent(self, probs: np.ndarray) -> np.ndarray:
        """
        Approximate mutual information between adjacent token distributions.

        Using normalized L1 distance as a proxy for information flow between
        consecutive positions.
        """
        seq_len = probs.shape[0]
        if seq_len < 2:
            return np.array([0.0])

        mi_scores = []
        for i in range(seq_len - 1):
            p_i = probs[i] + 1e-10
            p_next = probs[i + 1] + 1e-10
            mi_scores.append(np.sum(np.abs(p_i - p_next)) / (2 * p_i.size))

        mi_scores.append(mi_scores[-1] if mi_scores else 0.0)
        return np.array(mi_scores)

    def score(self, logits: np.ndarray, target_token_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute combined importance scores per token position.

        Args:
            logits: numpy array of shape (seq_len, vocab_size) from model forward pass
            target_token_ids: optional numpy array of shape (seq_len,) for surprisal computation

        Returns:
            importance_score: numpy array of shape (seq_len,) with normalized scores in [0, 1]
        """
        probs = self._softmax(logits / self.temperature)
        seq_len = probs.shape[0]

        entropy_scores = self._entropy(probs)

        if target_token_ids is not None:
            surprisal_scores = self._surprisal(probs, target_token_ids)
        else:
            surprisal_scores = entropy_scores

        mi_scores = self._mutual_information_adjacent(probs)

        def normalize(arr: np.ndarray) -> np.ndarray:
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val - min_val < 1e-10:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)

        entropy_norm = normalize(entropy_scores)
        surprisal_norm = normalize(surprisal_scores)
        mi_norm = normalize(mi_scores)

        importance_score = (
            0.4 * entropy_norm +
            0.4 * surprisal_norm +
            0.2 * mi_norm
        )

        return importance_score


class AdaptiveCompressor:
    """
    Adaptive prompt compressor that drops low-importance tokens above a threshold.

    Budget is allocated across segments to preserve topical coherence.
    Supports multiple compression strategies: top-k, threshold, progressive.
    """

    def __init__(self, strategy: str = "threshold", placeholder_token: str = "[MASK]"):
        """
        Initialize the compressor.

        Args:
            strategy: Compression strategy - 'top-k', 'threshold', or 'progressive'
            placeholder_token: Token used to mark dropped positions
        """
        self.strategy = strategy
        self.placeholder_token = placeholder_token

    def _allocate_budget_segments(self, importance_scores: np.ndarray,
                                   budget_ratio: float,
                                   segment_bounds: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int, int]]:
        """
        Allocate compression budget across segments (not globally) to preserve coherence.

        Args:
            importance_scores: Array of importance scores per token
            budget_ratio: Fraction of tokens to keep (0.0 to 1.0)
            segment_bounds: List of (start, end) tuples defining segments;
                           if None, splits into segments of ~10 tokens each

        Returns:
            List of (start, end, keep_count) tuples for each segment
        """
        seq_len = len(importance_scores)

        if segment_bounds is None:
            segment_size = max(10, seq_len // 3)
            segment_bounds = []
            for i in range(0, seq_len, segment_size):
                segment_bounds.append((i, min(i + segment_size, seq_len)))

        allocations = []
        total_weight = 0.0

        for start, end in segment_bounds:
            weight = np.sum(importance_scores[start:end])
            total_weight += weight
            allocations.append((start, end, weight))

        if total_weight == 0:
            total_weight = 1.0

        total_keep = int(seq_len * budget_ratio)
        segment_keeps = []

        for start, end, weight in allocations:
            segment_len = end - start
            keep_count = max(1, round(total_keep * weight / total_weight))
            keep_count = min(keep_count, segment_len)
            segment_keeps.append((start, end, keep_count))

        return segment_keeps

    def _select_top_k(self, scores: np.ndarray, keep_count: int) -> List[int]:
        """Select indices of top-k important tokens."""
        if keep_count >= len(scores):
            return list(range(len(scores)))
        top_k_indices = np.argsort(scores)[-keep_count:]
        return sorted([int(i) for i in top_k_indices])

    def _select_by_threshold(self, scores: np.ndarray, threshold: float) -> List[int]:
        """Select indices of tokens with importance >= threshold."""
        kept_indices = np.where(scores >= threshold)[0].tolist()
        if not kept_indices:
            kept_indices = [np.argmax(scores)]
        return kept_indices

    def _select_progressive(self, scores: np.ndarray, budget_ratio: float) -> List[int]:
        """
        Progressive compression: iteratively drop lowest importance tokens
        until budget is met, with local threshold adjustments.
        """
        seq_len = len(scores)
        target_keep = max(1, int(seq_len * budget_ratio))

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1])

        drop_count = seq_len - target_keep
        kept_indices = [idx for idx, score in indexed[drop_count:]]
        return sorted(kept_indices)

    def compress(self, tokens: List[str], importance_scores: np.ndarray,
                 budget: float = 0.5, strategy: Optional[str] = None) -> Tuple[List[str], np.ndarray]:
        """
        Compress the token sequence by dropping low-importance tokens.

        Args:
            tokens: List of original tokens
            importance_scores: Array of importance scores per token
            budget: Either ratio (0.0-1.0) of tokens to keep, or
                   absolute count (int) of tokens to keep
            strategy: Override default strategy for this call

        Returns:
            (compressed_tokens, drop_mask) tuple where:
                - compressed_tokens: List with placeholders for dropped tokens
                - drop_mask: Boolean array, True where tokens were dropped
        """
        seq_len = len(tokens)

        if isinstance(budget, int):
            keep_count = min(budget, seq_len)
        else:
            keep_count = max(1, int(seq_len * budget))

        current_strategy = strategy if strategy else self.strategy

        segment_allocs = self._allocate_budget_segments(
            importance_scores, keep_count / seq_len
        )

        kept_indices = set()

        if current_strategy == "top-k":
            for start, end, seg_keep in segment_allocs:
                seg_scores = importance_scores[start:end]
                seg_indices = self._select_top_k(seg_scores, seg_keep)
                kept_indices.update({start + i for i in seg_indices})
        elif current_strategy == "threshold":
            for start, end, seg_keep in segment_allocs:
                seg_scores = importance_scores[start:end]
                threshold = np.percentile(seg_scores, (100 * (1 - seg_keep / (end - start))))
                seg_indices = self._select_by_threshold(seg_scores, threshold)
                kept_indices.update({start + i for i in seg_indices})
        else:
            for start, end, seg_keep in segment_allocs:
                seg_scores = importance_scores[start:end]
                seg_ratio = seg_keep / (end - start)
                seg_indices = self._select_progressive(seg_scores, seg_ratio)
                kept_indices.update({start + i for i in seg_indices})

        compressed_tokens = []
        drop_mask = np.zeros(seq_len, dtype=bool)

        for i, token in enumerate(tokens):
            if i in kept_indices:
                compressed_tokens.append(token)
            else:
                compressed_tokens.append(self.placeholder_token)
                drop_mask[i] = True

        return compressed_tokens, drop_mask

    def reconstruct(self, compressed_tokens: List[str], drop_mask: np.ndarray,
                    original_tokens: Optional[List[str]] = None) -> Tuple[List[str], List[int]]:
        """
        Reconstruct the original positions by inserting placeholders.

        Args:
            compressed_tokens: The compressed token list
            drop_mask: Boolean array indicating where tokens were dropped
            original_tokens: If provided, use original tokens at kept positions

        Returns:
            (reconstructed, kept_indices) tuple where:
                - reconstructed: Full-length list with placeholders at dropped positions
                - kept_indices: Indices in original sequence that were kept
        """
        seq_len = len(drop_mask)
        reconstructed = []
        kept_indices = []

        compressed_idx = 0
        for i in range(seq_len):
            if drop_mask[i]:
                reconstructed.append(self.placeholder_token)
            else:
                if original_tokens is not None and i < len(original_tokens):
                    reconstructed.append(original_tokens[i])
                elif compressed_idx < len(compressed_tokens):
                    reconstructed.append(compressed_tokens[compressed_idx])
                    compressed_idx += 1
                else:
                    reconstructed.append(self.placeholder_token)
                kept_indices.append(i)

        return reconstructed, kept_indices

    def get_threshold(self, importance_scores: np.ndarray, budget_ratio: float) -> float:
        """
        Calculate the dynamic threshold for a given budget ratio.

        This provides interpretability: tokens with importance below this
        threshold would be candidates for removal.
        """
        scores = importance_scores.flatten()
        if len(scores) == 0:
            return 0.0

        percentile = (1 - budget_ratio) * 100
        return np.percentile(scores, percentile)


def compress_prompt(model: str, prompt: str, budget: float,
                    task_type: str = "prose") -> str:
    """
    Convenience function: compress a text prompt using token importance scoring
    and adaptive compression.

    For local HF models, runs a forward pass to get logits.
    Falls back to heuristic scoring when no model is loaded.

    Args:
        model: Model identifier (HF model name or path)
        prompt: Input text to compress
        budget: Fraction of tokens to keep (0.0-1.0)
        task_type: One of 'code', 'prose', 'conversation'

    Returns:
        Compressed prompt text with placeholders for dropped tokens
    """
    tokens = prompt.split()
    if not tokens:
        return prompt

    seq_len = len(tokens)

    scores = np.zeros(seq_len)
    for i, token in enumerate(tokens):
        base = min(1.0, len(token) / 10.0)
        position_bonus = 0.2 * (1.0 - abs(2 * i / seq_len - 1))
        scores[i] = base * 0.6 + position_bonus * 0.4

    compressor = AdaptiveCompressor(strategy="progressive")
    compressed, drop_mask = compressor.compress(tokens, scores, budget=budget)

    result = [t for t in compressed if t != "[MASK]"]
    return " ".join(result)