import json
import math

import mlx.core as mx
import pytest

from quantbench.quality import make_windows, perplexity
from quantbench.report import save_chart, to_frame
from quantbench.variants import parse_variant, parse_variants


# ---------------------------------------------------------------- variants
@pytest.mark.parametrize("spec,name,bits,group", [
    ("fp16", "fp16", None, 64),
    ("4", "q4", 4, 64),
    ("q8", "q8", 8, 64),
    ("4g32", "q4g32", 4, 32),
    ("mixed_3_6", "mixed_3_6", 3, 64),
])
def test_parse_variant(spec, name, bits, group):
    v = parse_variant(spec)
    assert (v.name, v.bits, v.group_size) == (name, bits, group)


def test_parse_variant_labels():
    assert parse_variant("fp16").label == "float16"
    assert parse_variant("4").label == "4-bit"
    assert parse_variant("4g32").label == "4-bit g32"
    assert parse_variant("mixed_3_6").label == "mixed 3/6-bit"


@pytest.mark.parametrize("bad", ["7", "4g16", "fp8", "mixed_9_9", ""])
def test_parse_variant_rejects(bad):
    with pytest.raises(ValueError):
        parse_variant(bad)


def test_parse_variants_rejects_duplicates():
    with pytest.raises(ValueError):
        parse_variants("4,q4")


# ---------------------------------------------------------------- perplexity
class UniformModel:
    """Predicts every token with equal probability, so perplexity == vocab size."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, x):
        return mx.zeros((*x.shape, self.vocab))


def test_make_windows_shapes():
    w = make_windows(list(range(1000)), seq_len=100, num_samples=5)
    assert w.shape == (5, 100)
    assert w[1, 0].item() == 100
    assert make_windows(list(range(250)), 100, 5).shape == (2, 100)
    with pytest.raises(ValueError):
        make_windows(list(range(50)), 100, 5)


def test_perplexity_uniform_equals_vocab():
    windows = make_windows(list(range(10)) * 40, seq_len=16, num_samples=10)
    ppl, se = perplexity(UniformModel(vocab=50), windows, batch_size=3)
    assert math.isclose(ppl, 50, rel_tol=1e-4)
    assert se == pytest.approx(0, abs=1e-3)


# ---------------------------------------------------------------- report
RESULTS = {
    "model": "test/model",
    "machine": {"chip": "Apple M1", "ram_gb": 8.6},
    "variants": [
        {"variant": "fp16", "label": "float16", "bits": None, "size_gb": 1.0, "bits_per_weight": 16.0,
         "ttft_ms": 50, "prompt_tps": 500, "gen_tps": 40.0, "peak_mem_gb": 1.1, "ppl": 10.0, "ppl_se": 0.1},
        {"variant": "q4", "label": "4-bit", "bits": 4, "size_gb": 0.25, "bits_per_weight": 4.5,
         "ttft_ms": 30, "prompt_tps": 600, "gen_tps": 100.0, "peak_mem_gb": 0.4, "ppl": 11.0, "ppl_se": 0.1},
    ],
}


def test_to_frame_relative_columns():
    df = to_frame(RESULTS)
    q4 = df[df.variant == "q4"].iloc[0]
    assert df.iloc[0]["is_baseline"] and not q4["is_baseline"]
    assert q4["size_x"] == pytest.approx(4.0)
    assert q4["speed_x"] == pytest.approx(2.5)
    assert q4["ppl_delta_pct"] == pytest.approx(10.0)


def test_save_chart_writes_png(tmp_path):
    out = save_chart(to_frame(RESULTS), RESULTS, tmp_path / "summary.png")
    assert out.exists() and out.stat().st_size > 1000


def test_report_without_speed_or_ppl(tmp_path):
    slim = json.loads(json.dumps(RESULTS))
    for v in slim["variants"]:
        for k in ("ttft_ms", "prompt_tps", "gen_tps", "peak_mem_gb", "ppl", "ppl_se"):
            v.pop(k)
    out = save_chart(to_frame(slim), slim, tmp_path / "s.png")
    assert out.exists()
