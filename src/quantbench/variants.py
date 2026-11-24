"""Quantization variants and how to parse them from the command line.

Spec syntax (comma-separated on the CLI):
    fp16 | bf16          unquantized, cast to that dtype
    8, 6, 4, 3, 2        uniform affine quantization at that many bits (group size 64)
    4g32                 same, with an explicit group size
    mixed_4_6 ...        mlx-lm mixed recipes: most layers low bits, sensitive layers 6 bits
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIXED_RECIPES = {"mixed_2_6": 2, "mixed_3_4": 3, "mixed_3_6": 3, "mixed_4_6": 4}
VALID_BITS = {2, 3, 4, 5, 6, 8}
DEFAULT_GROUP_SIZE = 64


@dataclass(frozen=True)
class Variant:
    name: str
    bits: int | None = None  # None means unquantized
    group_size: int = DEFAULT_GROUP_SIZE
    dtype: str | None = None  # for unquantized variants
    recipe: str | None = None  # mixed-precision recipe name

    @property
    def quantized(self) -> bool:
        return self.bits is not None

    @property
    def label(self) -> str:
        if self.recipe:
            return self.recipe.replace("mixed_", "mixed ").replace("_", "/") + "-bit"
        if not self.quantized:
            return self.dtype
        suffix = "" if self.group_size == DEFAULT_GROUP_SIZE else f" g{self.group_size}"
        return f"{self.bits}-bit{suffix}"


def parse_variant(spec: str) -> Variant:
    s = spec.strip().lower()
    if s in ("fp16", "float16", "16"):
        return Variant(name="fp16", dtype="float16")
    if s in ("bf16", "bfloat16"):
        return Variant(name="bf16", dtype="bfloat16")
    if s in MIXED_RECIPES:
        return Variant(name=s, bits=MIXED_RECIPES[s], recipe=s)

    m = re.fullmatch(r"q?(\d+)(?:g(\d+))?", s)
    if not m:
        raise ValueError(
            f"Unknown variant '{spec}'. Use fp16, bf16, a bit width like 4 or 4g32, "
            f"or one of {', '.join(MIXED_RECIPES)}."
        )
    bits = int(m.group(1))
    group = int(m.group(2)) if m.group(2) else DEFAULT_GROUP_SIZE
    if bits not in VALID_BITS:
        raise ValueError(f"Unsupported bit width {bits}; choose from {sorted(VALID_BITS)}.")
    if group not in (32, 64, 128):
        raise ValueError(f"Unsupported group size {group}; choose 32, 64 or 128.")
    name = f"q{bits}" if group == DEFAULT_GROUP_SIZE else f"q{bits}g{group}"
    return Variant(name=name, bits=bits, group_size=group)


def parse_variants(specs: str) -> list[Variant]:
    variants = [parse_variant(s) for s in specs.split(",") if s.strip()]
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate variants in '{specs}'.")
    if not variants:
        raise ValueError("No variants given.")
    return variants
