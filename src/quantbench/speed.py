"""Generation speed and memory: time to first token, prompt and decode tokens/sec, peak memory."""

from __future__ import annotations

import statistics
import time

DEFAULT_PROMPT = (
    "Explain, step by step, how a CPU cache works and why cache misses slow programs down. "
    "Use a concrete example with numbers."
)


def build_prompt(tokenizer, text: str):
    """Use the model's chat template when it has one, so instruct models behave normally."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True
        )
    return text


def generate_once(model, tokenizer, prompt, max_tokens: int) -> dict:
    import mlx.core as mx
    from mlx_lm import stream_generate

    mx.reset_peak_memory()
    start = time.perf_counter()
    ttft = None
    last = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if ttft is None:
            ttft = time.perf_counter() - start
        last = resp
    if last is None:
        raise RuntimeError("Model produced no tokens.")
    return {
        "ttft_ms": ttft * 1000,
        "prompt_tokens": last.prompt_tokens,
        "prompt_tps": last.prompt_tps,
        "gen_tokens": last.generation_tokens,
        "gen_tps": last.generation_tps,
        "peak_mem_gb": last.peak_memory,
    }


def measure_speed(model, tokenizer, prompt_text: str = DEFAULT_PROMPT,
                  max_tokens: int = 128, runs: int = 3, warmup: int = 1) -> dict:
    """Median of `runs` greedy generations after `warmup` discarded runs."""
    prompt = build_prompt(tokenizer, prompt_text)
    for _ in range(warmup):
        generate_once(model, tokenizer, prompt, max_tokens)
    samples = [generate_once(model, tokenizer, prompt, max_tokens) for _ in range(runs)]

    out = {k: statistics.median(s[k] for s in samples) for k in samples[0] if k != "peak_mem_gb"}
    out["peak_mem_gb"] = max(s["peak_mem_gb"] for s in samples)
    return out
