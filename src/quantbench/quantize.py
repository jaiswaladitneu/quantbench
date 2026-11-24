"""Create (and cache) a quantized copy of a Hugging Face model for each variant."""

from __future__ import annotations

import shutil
from pathlib import Path

from .variants import Variant


def model_slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def variant_path(models_dir: Path, model_id: str, variant: Variant) -> Path:
    return Path(models_dir) / f"{model_slug(model_id)}--{variant.name}"


def ensure_variant(model_id: str, variant: Variant, models_dir: Path) -> Path:
    """Convert `model_id` to `variant` unless a finished copy already exists.

    `model_id` should be an unquantized Hugging Face model (fp16/bf16 weights);
    quantizing an already-quantized model is not meaningful.
    """
    from mlx_lm import convert

    out = variant_path(models_dir, model_id, variant)
    if (out / "config.json").exists():
        return out

    # Write to a temp dir first so an interrupted conversion is never mistaken for a finished one.
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    out.parent.mkdir(parents=True, exist_ok=True)

    if variant.quantized:
        convert(
            model_id,
            mlx_path=str(tmp),
            quantize=True,
            q_bits=variant.bits,
            q_group_size=variant.group_size,
            quant_predicate=variant.recipe,
        )
    else:
        convert(model_id, mlx_path=str(tmp), dtype=variant.dtype)

    tmp.rename(out)
    return out


def disk_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1e9
