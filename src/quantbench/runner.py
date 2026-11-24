"""Run the full benchmark: quantize -> load -> measure -> save, one variant at a time."""

from __future__ import annotations

import gc
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .quantize import disk_size_gb, ensure_variant, model_slug
from .variants import Variant

console = Console()


@dataclass
class RunOptions:
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")
    prompt: str | None = None
    max_tokens: int = 128
    runs: int = 3
    seq_len: int = 512
    ppl_samples: int = 64
    batch_size: int = 4
    skip_speed: bool = False
    skip_ppl: bool = False


def machine_info() -> dict:
    import mlx.core as mx
    import mlx_lm
    import psutil

    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        chip = platform.processor()
    return {
        "chip": chip,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
    }


def results_file(results_dir: Path, model_id: str) -> Path:
    return Path(results_dir) / model_slug(model_id) / "results.json"


def _save(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_benchmark(model_id: str, variants: list[Variant], opts: RunOptions) -> Path:
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.utils import compute_bits_per_weight

    from .quality import load_eval_windows, perplexity
    from .speed import DEFAULT_PROMPT, measure_speed

    out_path = results_file(opts.results_dir, model_id)
    payload = {
        "model": model_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": machine_info(),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(opts).items()},
        "variants": [],
    }
    windows = None

    for i, v in enumerate(variants, 1):
        console.rule(f"[bold]{i}/{len(variants)}  {v.label}")
        path = ensure_variant(model_id, v, opts.models_dir)
        model, tokenizer = load(str(path))
        row = {
            "variant": v.name,
            "label": v.label,
            "bits": v.bits,
            "group_size": v.group_size if v.quantized else None,
            "size_gb": disk_size_gb(path),
            "bits_per_weight": compute_bits_per_weight(model),
        }

        if not opts.skip_speed:
            console.print("  measuring speed...")
            row |= measure_speed(model, tokenizer, opts.prompt or DEFAULT_PROMPT,
                                 opts.max_tokens, opts.runs)

        if not opts.skip_ppl:
            if windows is None:
                console.print("  loading WikiText-2 eval windows...")
                windows = load_eval_windows(tokenizer, opts.seq_len, opts.ppl_samples)
            console.print(f"  scoring perplexity on {windows.shape[0]} x {opts.seq_len} tokens...")
            row["ppl"], row["ppl_se"] = perplexity(model, windows, opts.batch_size)

        payload["variants"].append(row)
        _save(out_path, payload)  # save after every variant so a crash keeps earlier results
        console.print(f"  done: {_one_line(row)}")

        del model, tokenizer
        gc.collect()
        mx.clear_cache()

    return out_path


def _one_line(row: dict) -> str:
    parts = [f"{row['size_gb']:.2f} GB", f"{row['bits_per_weight']:.2f} bpw"]
    if "gen_tps" in row:
        parts.append(f"{row['gen_tps']:.1f} tok/s")
        parts.append(f"{row['peak_mem_gb']:.2f} GB peak")
    if "ppl" in row:
        parts.append(f"ppl {row['ppl']:.2f}")
    return ", ".join(parts)
