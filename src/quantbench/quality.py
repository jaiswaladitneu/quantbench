"""Quality: perplexity on a fixed slice of WikiText-2 (test split).

Every variant of a model shares the same tokenizer, so the same token windows are
scored for each one; differences in perplexity come from the weights alone.
"""

from __future__ import annotations

import math

WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1", "test")


def load_wikitext_text() -> str:
    from datasets import load_dataset

    path, name, split = WIKITEXT
    ds = load_dataset(path, name, split=split)
    return "\n\n".join(t for t in ds["text"] if t.strip())


def make_windows(tokens: list[int], seq_len: int, num_samples: int):
    """Cut the first num_samples * seq_len tokens into non-overlapping windows."""
    import mlx.core as mx

    n = min(num_samples, len(tokens) // seq_len)
    if n == 0:
        raise ValueError(f"Need at least {seq_len} tokens, got {len(tokens)}.")
    return mx.array(tokens[: n * seq_len]).reshape(n, seq_len)


def load_eval_windows(tokenizer, seq_len: int = 512, num_samples: int = 64):
    from transformers.utils import logging

    logging.set_verbosity_error()  # silence the harmless "sequence longer than max length" warning
    tokens = tokenizer.encode(load_wikitext_text())
    return make_windows(tokens, seq_len, num_samples)


def perplexity(model, windows, batch_size: int = 4) -> tuple[float, float]:
    """Return (perplexity, standard error) of next-token prediction over `windows`."""
    import mlx.core as mx
    import mlx.nn as nn

    losses = []
    for s in range(0, windows.shape[0], batch_size):
        batch = windows[s : s + batch_size]
        logits = model(batch[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        mx.eval(loss)
        losses.append(loss.flatten())

    all_losses = mx.concatenate(losses)
    mean = all_losses.mean().item()
    ppl = math.exp(mean)
    std = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    se = ppl * std / math.sqrt(all_losses.size)  # delta method
    return ppl, se
