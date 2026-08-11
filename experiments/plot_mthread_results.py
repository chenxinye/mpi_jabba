"""
Regenerate multithreading figures from an existing CSV.

Example:
python plot_mthread_results.py \
  --csv results_mthread/univariate_thread_scaling.csv \
  --outdir results_mthread \
  --formats pdf png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multithreading import plot_breakdown, plot_overview  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate multithreading figures from saved CSV results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "results_mthread" / "univariate_thread_scaling.csv",
        help="Path to univariate_thread_scaling.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for regenerated figures. Defaults to the CSV directory.",
    )
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--skip-overview", action="store_true", help="Do not regenerate univariate_scaling_overview.")
    parser.add_argument("--skip-breakdown", action="store_true", help="Do not regenerate univariate_jabba_breakdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = resolve_path(args.csv)
    outdir = resolve_path(args.outdir) if args.outdir is not None else csv_path.parent

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.skip_overview:
        plot_overview(df, outdir, args.fontsize, args.formats)

    phase_columns = {"compression_time_mean", "digitization_time_mean", "inverse_time_mean"}
    if not args.skip_breakdown:
        if phase_columns.issubset(df.columns):
            plot_breakdown(df, outdir, args.fontsize, args.formats)
        else:
            missing = ", ".join(sorted(phase_columns.difference(df.columns)))
            print(f"Skipping breakdown figure; missing columns: {missing}")

    print(f"Figures regenerated in: {outdir}")
    if not args.skip_overview:
        for fmt in args.formats:
            print(f"Overview figure: {outdir / f'univariate_scaling_overview.{fmt}'}")
    if not args.skip_breakdown and phase_columns.issubset(df.columns):
        for fmt in args.formats:
            print(f"Breakdown figure: {outdir / f'univariate_jabba_breakdown.{fmt}'}")


if __name__ == "__main__":
    main()
