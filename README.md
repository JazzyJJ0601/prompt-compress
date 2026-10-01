# Prompt-Compress

**Cut a long prompt in half and keep most of what it tells the model: keep the recent text, plus the older 16-token spans the model found most surprising.**

Measured on Qwen3-8B: with half of a 384-token context removed, the model predicts the next 64 tokens at **9.09** perplexity, against 9.76 for the best simple baseline (keep the most recent half) and 8.51 with the whole context. No training; the scores come from one forward pass of the model itself.

## Results (Qwen3-8B, WikiText-2 test, 40 passages)

Each passage is 384 tokens of context followed by 64 tokens to predict. The context is compressed, then we measure the model's perplexity on the 64 tokens that follow. Lower is better.

| What the model sees | 50% of context kept | 25% kept |
|---|---|---|
| Whole context | 8.51 | 8.51 |
| Nothing (floor) | 30.68 | 30.68 |
| Random tokens | 13.34 | 19.66 |
| Most surprising single tokens | 12.30 | 17.06 |
| Entropy + surprisal + MI score (this repo's first scorer) | 12.67 | 16.74 |
| Most recent tokens only | 9.76 | 11.19 |
| **Recent + most surprising spans** | **9.09** | **10.42** |

The recent/older split was chosen on 20 separate dev passages (75% recent at the 50% budget, 50% recent at 25%), then reported on the 40 test passages above, so the table is not tuned on its own test set. Full numbers for every split, dev and test: [`results/real.json`](results/real.json).

## Why it works

Two findings from the runs:

1. **Single-token dropping breaks text.** Keeping the individually most informative tokens leaves a word salad the model reads poorly; it only just beats random dropping. Keeping whole 16-token spans keeps phrases intact.
2. **Recency is most of the signal, but not all.** The nearest text matters most for what comes next, so a recent window is a strong baseline. Spending a quarter to a half of the budget on older spans with high average surprisal (names, numbers, new facts the model could not have guessed) beats spending it on more recent text.

## Honest notes

- **The first version of this repo was wrong in three ways**, which produced the old "50-100x worse" result. (1) The scorer read each token's surprisal from the wrong position (off by one). (2) The benchmark rebuilt the compressed prompt by joining raw tokenizer pieces like `Ġquick` with spaces, so the model was fed garbled text. (3) It measured the perplexity of the compressed text itself, which says nothing about whether the meaning survived. All three are fixed; token ids are now passed to the model directly.
- Perplexity on WikiText continuations is a proxy. Question answering over long documents (where compression is usually used) is not tested yet.
- One model, one seed, 40 test passages.

## Usage

```python
import numpy as np
from src.prompt_compress.core import span_pick

# surprisal[t] = -log p(token t | tokens before it), from one forward pass
keep = span_pick(surprisal, k=len(ids) // 2, recent=int(len(ids) // 2 * 0.75), span=16)
compressed_ids = ids[keep]
```

Reproduce (needs a local Qwen3-8B and about 17 GB of GPU memory):

```bash
python results/run_real.py
```

## Tests

```bash
python -m pytest tests/
```
