# Prompt-Compress

Dynamic prompt compression via information-theoretic token importance scoring. Reduces prompt length while preserving semantic content — no training, no fine-tuning, works with any decoder-only model.

## What it does

Prompt-Compress assigns importance scores to each token using the model's own logit distribution, then drops low-importance tokens. The scoring uses three information-theoretic measures:

- **Entropy** — high-uncertainty tokens carry more predictive value
- **Surprisal** — low-probability tokens carry more information content
- **Mutual information (adjacent)** — measures information flow between consecutive positions

Compression is adaptive: budget is allocated across segments to preserve topical coherence.

## Architecture

```
TokenImportanceScorer        AdaptiveCompressor
  ├── _softmax                 ├── _allocate_budget_segments
  ├── _entropy                 ├── _select_top_k
  ├── _surprisal               ├── _select_by_threshold
  ├── _mutual_information      ├── _select_progressive
  └── score (combined)         ├── compress
                               ├── reconstruct
                               └── get_threshold
```

Three compression strategies:
- **top-k** — keep the N most important tokens per segment
- **threshold** — keep tokens above a dynamic importance threshold
- **progressive** — iteratively drop lowest-importance tokens

## Usage

```python
from prompt_compress.core import compress_prompt

compressed = compress_prompt(
    model="gpt2",
    prompt="Your long prompt text here...",
    budget=0.6,
    task_type="prose"
)
```

Or from the command line:
```
prompt-compress --model gpt2 --prompt "Your text" --budget 0.6 --task-type prose
```

## How it compares

| Approach | Gradient-Free | Local HF | Task-Aware | Budget Control |
|----------|---------------|----------|------------|----------------|
| Prompt-Compress | Yes | Yes | Yes | Yes |
| LLMLingua | No (fine-tuning) | Partial | No | Limited |
| Selective Context | Yes | Yes | No | Fixed |
| ICAE | No (encoder) | Partial | No | Fixed |

## Key advantages

- **Zero training** — uses model's native logits directly
- **Universal** — any decoder-only HF model
- **Adaptive** — thresholds per-task and per-budget
- **Interpretable** — retention decisions based on measurable information metrics

## Requirements

- Python 3.10+
- NumPy

Optional for full model inference: `transformers`, `torch`.