# Prompt-Compress: measured results

Qwen3-8B, bf16. WikiText-2 test: 384-token context, then perplexity on the next 64 tokens.
Passages 0-39 = test (below), 40-59 = dev (used only to pick the recent share).
Command: `python results/run_real.py` (writes `results/real.json`).

| Method | 50% kept | 25% kept |
|---|---|---|
| Full context | 8.51 | 8.51 |
| No context | 30.68 | 30.68 |
| Random | 13.34 | 19.66 |
| Surprisal, single tokens | 12.30 | 17.06 |
| Combined score, single tokens | 12.67 | 16.74 |
| Recent only | 9.76 | 11.19 |
| Recent + surprising 16-token spans (recent share picked on dev: 0.75 / 0.5) | **9.09** | **10.42** |

`results/real_v1.json` is the first run of this benchmark (before the span method). Earlier results in this file (self-perplexity of 3 short garbled prompts) are withdrawn; see the README.
