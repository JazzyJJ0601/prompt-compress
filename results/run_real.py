#!/usr/bin/env python3
"""Real results: does a compressed context still help Qwen3-8B predict what comes next?

For each of N passages from WikiText-2 test: a CTX-token context followed by a
CONT-token continuation. The context is compressed by dropping tokens (token
ids are kept directly; no detokenise/retokenise round trip), then we measure
the model's perplexity on the continuation given the compressed context.

  full          whole context (the ceiling)
  none          no context at all (the floor)
  recent        keep the last k context tokens
  random        keep k random tokens, in order
  surprisal     keep the k most surprising tokens (selective-context rule)
  combined      keep the k best by this repo's entropy+surprisal+MI score

Lower continuation perplexity is better. Results go to results/real.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.prompt_compress.core import TokenImportanceScorer, span_pick  # noqa: E402

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
CTX, CONT, N = 384, 64, 40
DEV = 20  # passages after the test ones, used only to choose the recent fraction
SPAN = 16
RECENT_FRACS = [0.5, 0.75, 0.9]
KEEPS = [0.5, 0.25]
OUT = HERE / "real.json"


@torch.no_grad()
def cont_nll(model, ctx_ids, cont_ids):
    """Summed NLL of cont_ids given ctx_ids (both 1-D LongTensors)."""
    ids = torch.cat([ctx_ids, cont_ids]).unsqueeze(0).cuda()
    logits = model(ids).logits[0].float()
    start = len(ctx_ids)
    pred = logits[start - 1:-1] if start > 0 else logits[:-1]
    tgt = ids[0, start:] if start > 0 else ids[0, 1:]
    return torch.nn.functional.cross_entropy(pred, tgt, reduction="sum").item(), len(tgt)


def keep_top(scores, k):
    return np.sort(np.argsort(-scores, kind="stable")[:k])


def main():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, local_files_only=True, device_map="cuda").eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    span = CTX + CONT
    rng = np.random.default_rng(0)
    scorer = TokenImportanceScorer()

    totals = {}
    def add(name, nll_n):
        a = totals.setdefault(name, [0.0, 0])
        a[0] += nll_n[0]
        a[1] += nll_n[1]

    for i in range(N + DEV):
        split = "test" if i < N else "dev"
        chunk = ids[i * span:(i + 1) * span]
        ctx, cont = chunk[:CTX], chunk[CTX:]
        with torch.no_grad():
            logits = model(ctx.unsqueeze(0).cuda()).logits[0].float()
        logp = torch.log_softmax(logits, -1)
        surpr = np.empty(CTX)
        surpr[1:] = (-logp[:-1].gather(1, ctx[1:].cuda().unsqueeze(1)).squeeze(1)).cpu().numpy()
        surpr[0] = surpr[1:].max()
        combined = scorer.score(logits.cpu().numpy(), ctx.numpy())

        add(f"{split}/full", cont_nll(model, ctx, cont))
        add(f"{split}/none", cont_nll(model, ctx[:1], cont))  # keep only the first token so the model has a start
        for keep in KEEPS:
            k = int(CTX * keep)
            picks = {
                "recent": np.arange(CTX - k, CTX),
                "random": np.sort(rng.choice(CTX, k, replace=False)),
                "surprisal": keep_top(surpr, k),
                "combined": keep_top(combined, k),
            }
            for frac in RECENT_FRACS:
                picks[f"recent+spans{frac}"] = span_pick(surpr, k, int(k * frac), SPAN)
            for name, idx in picks.items():
                add(f"{split}/{name}@{keep}", cont_nll(model, ctx[torch.as_tensor(idx)], cont))
        print(f"passage {i + 1}/{N + DEV}", flush=True)

    results = {"setup": {"model": "Qwen3-8B", "data": f"WikiText-2 test: passages 0-{N - 1} = test, {N}-{N + DEV - 1} = dev (dev only chooses the recent fraction)",
                         "context_tokens": CTX, "continuation_tokens": CONT}}
    for name, (nll, n) in totals.items():
        results[name] = {"continuation_ppl": round(float(np.exp(nll / n)), 3)}
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
