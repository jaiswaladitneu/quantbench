"""Command-line interface: `quantbench run` and `quantbench report`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .variants import parse_variants

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Benchmark LLM quantization on Apple Silicon with MLX.")
console = Console()

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_VARIANTS = "fp16,8,6,4,3,mixed_3_6"


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Unquantized Hugging Face model id or local path.")] = DEFAULT_MODEL,
    variants: Annotated[str, typer.Option(help="Comma-separated: fp16, bf16, 8, 6, 4, 3, 2, 4g32, mixed_3_6 ...")] = DEFAULT_VARIANTS,
    models_dir: Annotated[Path, typer.Option(help="Where converted models are cached.")] = Path("models"),
    results_dir: Annotated[Path, typer.Option(help="Where results are written.")] = Path("results"),
    prompt: Annotated[str | None, typer.Option(help="Prompt for the speed test.")] = None,
    max_tokens: Annotated[int, typer.Option(help="Tokens generated per speed run.")] = 128,
    runs: Annotated[int, typer.Option(help="Timed speed runs per variant (median reported).")] = 3,
    seq_len: Annotated[int, typer.Option(help="Tokens per perplexity window.")] = 512,
    ppl_samples: Annotated[int, typer.Option(help="Number of perplexity windows.")] = 64,
    batch_size: Annotated[int, typer.Option(help="Perplexity batch size (lower it if memory is tight).")] = 4,
    skip_speed: Annotated[bool, typer.Option(help="Skip the speed/memory test.")] = False,
    skip_ppl: Annotated[bool, typer.Option(help="Skip the perplexity test.")] = False,
):
    """Quantize MODEL into each variant, benchmark them all, and print the comparison."""
    from .report import write_report
    from .runner import RunOptions, run_benchmark

    try:
        vs = parse_variants(variants)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--variants")

    opts = RunOptions(models_dir, results_dir, prompt, max_tokens, runs,
                      seq_len, ppl_samples, batch_size, skip_speed, skip_ppl)
    path = run_benchmark(model, vs, opts)
    csv, chart = write_report(path, console)
    console.print(f"\nSaved [bold]{path}[/], [bold]{csv}[/] and [bold]{chart}[/]")


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Path to a results.json written by `run`.")],
):
    """Re-print the table and re-draw the chart from saved results."""
    from .report import write_report

    csv, chart = write_report(results, console)
    console.print(f"\nSaved [bold]{csv}[/] and [bold]{chart}[/]")


if __name__ == "__main__":
    app()
