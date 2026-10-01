# Prompt-Compress Real Benchmark Results

**Status:** Negative result on Qwen3-8B: dropping half the prompt tokens raised perplexity 50-100x (e.g. 37 to 4,276). The current importance scorer does not preserve meaning.

## Command
python3 repos/prompt-compress/results/run_real.py

## Results

| Prompt | Original Tokens | Retained Tokens | Baseline PPL | Compressed PPL |
|--------|-----------------|-----------------|----------------|----------------|
| The quick brown fox jumps over... | 10 | 5 | 3.46 | 186.56 |
| Artificial intelligence is tra... | 8 | 4 | 51.12 | 1902.60 |
| Machine learning models need l... | 9 | 4 | 37.10 | 4276.45 |

## Interpretation

The compression method scores tokens by information-theoretic importance and drops the lowest-scoring half. As expected, compressing to 50% of tokens increases perplexity because semantic information is lost when tokens are removed. The increase is large, so the current scorer is removing tokens the model needs.
