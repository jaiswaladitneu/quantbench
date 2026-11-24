"""Turn results.json into a comparison table, a CSV and a chart."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

# Reference palette (light mode) from the project's chart guidelines.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"
BASELINE = "#86b6ef"  # lighter step of the same hue marks the unquantized reference


def load_results(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def to_frame(results: dict) -> pd.DataFrame:
    """One row per variant, plus columns relative to the baseline (first unquantized, else first row)."""
    df = pd.DataFrame(results["variants"])
    unq = df.index[df["bits"].isna()]
    base = df.loc[unq[0] if len(unq) else df.index[0]]

    df["is_baseline"] = df.index == base.name
    df["size_x"] = base["size_gb"] / df["size_gb"]
    if "gen_tps" in df:
        df["speed_x"] = df["gen_tps"] / base["gen_tps"]
    if "ppl" in df:
        df["ppl_delta_pct"] = (df["ppl"] / base["ppl"] - 1) * 100
    return df


def print_table(df: pd.DataFrame, results: dict, console: Console | None = None):
    console = console or Console()
    m = results.get("machine", {})
    t = Table(title=f"{results['model']}  ·  {m.get('chip', '?')}, {m.get('ram_gb', '?')} GB",
              title_justify="left", box=box.SIMPLE_HEAD, pad_edge=False, padding=(0, 1), collapse_padding=True)
    has_speed, has_ppl = "gen_tps" in df, "ppl" in df
    cols = ["Variant", "Size GB", "bpw"]
    if has_speed:
        cols += ["TTFT ms", "In t/s", "Out t/s", "Speed", "Peak GB"]
    if has_ppl:
        cols += ["PPL", "ΔPPL"]
    for c in cols:
        t.add_column(c, justify="left" if c == "Variant" else "right",
                     min_width=8 if c == "Variant" else 4)

    for _, r in df.iterrows():
        base = r["is_baseline"]
        cells = [f"{r['label']}*" if base else r["label"],
                 f"{r['size_gb']:.2f}", f"{r['bits_per_weight']:.2f}"]
        if has_speed:
            cells += [f"{r['ttft_ms']:.0f}", f"{r['prompt_tps']:.0f}", f"{r['gen_tps']:.1f}",
                      f"{r['speed_x']:.2f}x", f"{r['peak_mem_gb']:.2f}"]
        if has_ppl:
            cells += [f"{r['ppl']:.2f}", "—" if base else f"{r['ppl_delta_pct']:+.1f}%"]
        t.add_row(*cells)
    console.print(t)
    console.print("[dim]* baseline. In/Out t/s = prompt / generation tokens per sec (median). "
                  "Speed = Out t/s vs baseline. PPL = WikiText-2 perplexity, lower is better.[/]")


def _style(ax, title: str, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, df, col, fmt, title, ylabel):
    colors = [BASELINE if b else SERIES for b in df["is_baseline"]]
    bars = ax.bar(df["label"], df[col], color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, df[col]):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color=INK_2)
    _style(ax, title, ylabel)
    ax.tick_params(axis="x", rotation=30)


def save_chart(df: pd.DataFrame, results: dict, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["size"] + (["speed", "mem"] if "gen_tps" in df else []) + (["ppl"] if "ppl" in df else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), facecolor=SURFACE)
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, p in zip(axes, panels):
        if p == "size":
            _bars(ax, df, "size_gb", "{:.2f}", "Size on disk", "GB")
        elif p == "speed":
            _bars(ax, df, "gen_tps", "{:.0f}", "Generation speed", "tokens / sec")
        elif p == "mem":
            _bars(ax, df, "peak_mem_gb", "{:.2f}", "Peak memory", "GB")
        elif p == "ppl":
            # Quality vs. size trade-off: lower-left is better.
            ax.errorbar(df["size_gb"], df["ppl"], yerr=df["ppl_se"], fmt="none",
                        ecolor=AXIS, elinewidth=1, capsize=0, zorder=1)
            ax.scatter(df["size_gb"], df["ppl"], s=64, zorder=2, edgecolors=SURFACE, linewidths=2,
                       c=[BASELINE if b else SERIES for b in df["is_baseline"]])
            for _, r in df.iterrows():
                ax.annotate(r["label"], (r["size_gb"], r["ppl"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=8, color=INK_2)
            _style(ax, "Perplexity vs size (lower is better)", "perplexity")
            ax.set_xlabel("size on disk (GB)", color=INK_2, fontsize=9)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.margins(x=0.15, y=0.1)

    fig.suptitle(f"{results['model']}  ·  {results.get('machine', {}).get('chip', '')}",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_report(results_path: Path, console: Console | None = None) -> tuple[Path, Path]:
    results = load_results(results_path)
    df = to_frame(results)
    print_table(df, results, console)
    out_dir = Path(results_path).parent
    csv = out_dir / "results.csv"
    df.to_csv(csv, index=False)
    chart = save_chart(df, results, out_dir / "summary.png")
    return csv, chart
