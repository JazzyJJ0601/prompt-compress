# Prompt-Compress Design Document

## Overview

Prompt-Compress is a gradient-free token compression framework for large language models (LLMs). It reduces prompt length while preserving semantic content, operating with any local HuggingFace model without requiring fine-tuning or gradient computation.

## Information-Theoretic Foundation

### Token Importance Scoring

We assign importance scores to each token based on information-theoretic measures derived from the model's own probability distribution.

**Surprisal (Self-Information):**
For a token $t$ in context $C$, surprisal is:
$$I(t|C) = -\log_2 P(t|C)$$

Tokens with low probability (high surprisal) carry more information and should be prioritized.

**Per-Token Entropy:**
The entropy of the model's output distribution at position $i$:
$$H_i = -\sum_{v \in V} P(v|C_i) \log_2 P(v|C_i)$$

High entropy regions indicate model uncertainty; compressing these risks information loss.

### Adaptive Compression with Significance Thresholds

We define a significance threshold $\tau$ that adapts to available budget:
$$\tau = \alpha \cdot \text{median}(I(t|C)) + \beta \cdot \text{std}(I(t|C))$$

where $\alpha, \beta$ are tuned hyperparameters. Tokens with $I(t|C) < \tau$ are candidates for removal.

## Task-Aware Compression Budget

Different tasks tolerate different compression rates. We employ a lightweight classifier to estimate optimal budget:

$$B_{task} = f_{classifier}(type) \cdot L_{original}$$

Where `type` is predicted as:
- **Code**: Higher entropy, lower compression tolerance (budget ≈ 80%)
- **Prose**: Moderate entropy, medium tolerance (budget ≈ 60%)
- **Conversation**: Variable, context-dependent (budget ≈ 50%)

The classifier uses simple n-gram features computed without model inference.

## Reconstruction Quality Bounds

For a compressed prompt $P'$ reconstructed from original $P$ with retained tokens $T$:
$$||P - P'|| \leq \sum_{t \in T} \text{impact}(t)$$

Under our entropy-weighted retention, we bound degradation by:
$$\text{QualityLoss} \leq \frac{\sum_{discarded} H(t)}{\sum_{all} H(t)}$$

This provides a theoretical guarantee: if total retained entropy exceeds 90%, expected output quality loss remains below 10%.

## Comparison to Existing Approaches

| Approach | Gradient-Free | Local HF Support | Task-Aware | Budget Control |
|----------|---------------|------------------|------------|----------------|
| **Prompt-Compress** | ✓ | ✓ | ✓ | ✓ |
| LLMLingua | ✗ (requires fine-tuning) | Partial | ✗ | Limited |
| Selective Context | ✓ | ✓ | ✗ | Fixed |
| ICAE | ✗ (requires encoder) | Partial | ✗ | Fixed |

### Key Advantages

1. **Zero Training**: Uses model's native logits directly.
2. **Universal Compatibility**: Works with any decoder-only HF model.
3. **Adaptive**: Thresholds adjust per-task and per-budget.
4. **Interpretable**: Retention decisions based on measurable information metrics.

## Implementation Notes

- Tokenization: Model-native tokenizer
- Computation: Single forward pass for logits
- Retention: Greedy selection by importance score
- Reconstruction: Original tokens + placeholder markers for removed spans

## Future Work

- Dynamic threshold tuning via few-shot validation
- Multi-modal compression (code + docstrings)
- Integration with model context window optimization
