"""
Measure compression boundary artifacts under domain decomposition.

This experiment answers the reviewer question:

    for the compression, domain decomposition introduces boundary artifacts
    which might slightly increase the number of pieces. It would be highly
    beneficial to include a small empirical metric showing how much the number
    of pieces (N) increases as the number of partitions (M) grows.

Run from /Users/chenxinye/mpi_jabba/experiments:

    python compression_boundary_pieces.py \
      --lengths 200000 \
      --partitions 1 2 4 8 16 32 \
      --tol 0.5 \
      --kind random_walk \
      --outdir results_boundary_pieces \
      --formats pdf png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from jabba import JABBA
import jabba.compression as jabba_compression


def make_series(n: int, seed: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.normal(0.0, 1.0, n).astype(np.float64)
    if kind == "random_walk":
        return np.cumsum(rng.normal(0.0, 1.0, n)).astype(np.float64)
    if kind == "seasonal":
        x = np.linspace(0.0, 80.0 * np.pi, n)
        signal = np.sin(x) + 0.5 * np.sin(0.2 * x + 0.5) + 0.15 * np.sin(x / 30.0)
        return (signal + 0.35 * rng.normal(size=n)).astype(np.float64)
    raise ValueError("kind must be one of: gaussian, random_walk, seasonal")


def make_model(tol: float, partitions: int) -> JABBA:
    return JABBA(
        tol=tol,
        init="agg",
        alpha=0.05,
        sorting="norm",
        auto_digitize=False,
        partition=partitions,
        verbose=0,
        prefer_mpi=False,
        center_kind="seed",
    )


def compress_pieces(series: np.ndarray, tol: float, partitions: int, n_jobs: int) -> tuple[list[np.ndarray], float, int]:
    model = make_model(tol, partitions)
    t0 = time.perf_counter()
    pieces = model.parallel_compress(series, n_jobs=max(1, int(n_jobs)))
    elapsed = time.perf_counter() - t0
    return [np.asarray(block, dtype=np.float64) for block in pieces], float(elapsed), int(len(pieces))


def count_pieces(pieces: list[np.ndarray]) -> int:
    return int(sum(len(block) for block in pieces))


def split_boundaries(n: int, partitions: int) -> list[int]:
    partitions = max(1, min(int(partitions), max(1, n // 2)))
    interval = int(n / partitions)
    if interval <= 0:
        return []
    return [i * interval for i in range(1, partitions)]


def boundary_forced_extra(serial_pieces: np.ndarray, n: int, partitions: int) -> int:
    """Count serial compressed pieces that cross partition boundaries.

    If the serial piece sequence is kept fixed and a partition boundary falls
    strictly inside a serial piece, that piece must be split into two local
    pieces. Since each boundary can lie inside at most one serial piece, this
    metric is bounded by M-1 and directly measures the boundary-induced piece
    increase from domain decomposition.
    """
    if len(serial_pieces) == 0:
        return 0
    lengths = np.maximum(np.rint(np.asarray(serial_pieces)[:, 0]).astype(int), 1)
    ends = np.cumsum(lengths)
    starts = np.concatenate(([0], ends[:-1]))
    extra = 0
    for boundary in split_boundaries(n, partitions):
        if np.any((starts < boundary) & (boundary < ends)):
            extra += 1
    return int(extra)


def summarize_repeats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "length",
        "kind",
        "seed",
        "tol",
        "partitions",
        "n_jobs",
        "actual_blocks",
        "serial_pieces",
        "forced_parallel_pieces",
        "forced_extra_pieces",
        "observed_parallel_pieces",
        "observed_piece_change",
        "theoretical_max_extra",
        "forced_extra_ratio_percent",
        "forced_extra_per_boundary",
        "observed_change_percent",
        "compression_time_mean",
        "compression_time_std",
        "compression_time_min",
        "compression_time_max",
        "compression_backend",
        "warmup",
        "repeats",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex_table(path: Path, rows: list[dict[str, object]], length: int) -> None:
    selected = [row for row in rows if int(row["length"]) == int(length)]
    lines = [
        "\\begin{tabular}{r r r r r}",
        "\\toprule",
        "$M$ & Forced pieces & Extra pieces & Extra (\\%) & Bound $M-1$ \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{int(row['partitions'])} & "
            f"{int(row['forced_parallel_pieces']):,} & "
            f"{int(row['forced_extra_pieces']):,} & "
            f"{float(row['forced_extra_ratio_percent']):.4f} & "
            f"{int(row['theoretical_max_extra']):,} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(rows: list[dict[str, object]], outdir: Path, formats: list[str], fontsize: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    lengths = sorted({int(row["length"]) for row in rows})
    fig, axes = plt.subplots(1, len(lengths), figsize=(5.2 * len(lengths), 3.25), squeeze=False)

    for ax, length in zip(axes[0], lengths):
        subset = sorted([row for row in rows if int(row["length"]) == length], key=lambda row: int(row["partitions"]))
        partitions = np.asarray([int(row["partitions"]) for row in subset], dtype=int)
        extra = np.asarray([int(row["forced_extra_pieces"]) for row in subset], dtype=float)
        ratio = np.asarray([float(row["forced_extra_ratio_percent"]) for row in subset], dtype=float)
        bound = np.asarray([int(row["theoretical_max_extra"]) for row in subset], dtype=float)

        bars = ax.bar(
            np.arange(len(partitions)),
            extra,
            color="#4C78A8",
            edgecolor="#26364A",
            linewidth=0.8,
            label="boundary-forced extra pieces",
        )
        ax.plot(np.arange(len(partitions)), bound, color="#D55E00", marker="o", linewidth=2.0, label="bound $M-1$")
        ax.set_xticks(np.arange(len(partitions)))
        ax.set_xticklabels([str(p) for p in partitions])
        ax.set_xlabel("partitions $M$", fontsize=fontsize)
        ax.set_ylabel("boundary-forced extra pieces", fontsize=fontsize)
        ax.set_title(f"$n={length:,}$", fontsize=fontsize + 1)
        ax.tick_params(axis="both", labelsize=fontsize - 1)
        ax.grid(axis="y", alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(partitions)),
            ratio,
            color="#009E73",
            marker="s",
            mfc="white",
            mec="#009E73",
            linewidth=2.0,
            label="extra ratio",
        )
        ax2.set_ylabel("extra ratio (%)", fontsize=fontsize)
        ax2.tick_params(axis="y", labelsize=fontsize - 1)

        for rect, value in zip(bars, extra):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                f"{int(value)}",
                ha="center",
                va="bottom",
                fontsize=max(fontsize - 3, 6),
            )

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=fontsize - 1, loc="upper left", frameon=True)

    fig.tight_layout()
    for fmt in formats:
        path = outdir / f"compression_boundary_pieces.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure: {path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[200000])
    parser.add_argument("--partitions", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--kind", choices=["gaussian", "random_walk", "seasonal"], default="random_walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compression-backend", choices=["cython", "legacy-cython", "python"], default="cython")
    parser.add_argument("--outdir", type=Path, default=Path("results_boundary_pieces"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--fontsize", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    jabba_compression.set_compression_backend(args.compression_backend)

    rows: list[dict[str, object]] = []
    metadata = {
        "lengths": args.lengths,
        "partitions": args.partitions,
        "tol": args.tol,
        "kind": args.kind,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "requested_compression_backend": args.compression_backend,
        "compression_backend": jabba_compression.compression_backend(),
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for length in args.lengths:
        series = make_series(length, args.seed, args.kind)
        print(f"\n=== length={length:,} tol={args.tol:g} kind={args.kind} ===")

        for _ in range(args.warmup):
            compress_pieces(series, args.tol, 1, 1)
        serial_times: list[float] = []
        serial_piece_count = None
        serial_piece_blocks = None
        for repeat_idx in range(args.repeats):
            piece_blocks, elapsed, _ = compress_pieces(series, args.tol, 1, 1)
            serial_piece_count = count_pieces(piece_blocks)
            serial_piece_blocks = piece_blocks
            serial_times.append(elapsed)
            print(f"  serial repeat {repeat_idx + 1}/{args.repeats}: pieces={serial_piece_count:,} time={elapsed:.4f}s")
        assert serial_piece_count is not None
        assert serial_piece_blocks is not None
        serial_flat = np.vstack(serial_piece_blocks) if serial_piece_blocks else np.empty((0, 3), dtype=np.float64)

        for partitions in args.partitions:
            partitions = max(1, min(int(partitions), int(length // 2)))
            for _ in range(args.warmup):
                compress_pieces(series, args.tol, partitions, partitions)
            times: list[float] = []
            observed_piece_count = None
            actual_blocks = None
            for repeat_idx in range(args.repeats):
                piece_blocks, elapsed, blocks = compress_pieces(series, args.tol, partitions, partitions)
                observed_piece_count = count_pieces(piece_blocks)
                actual_blocks = blocks
                times.append(elapsed)
                observed_change = observed_piece_count - serial_piece_count
                print(
                    f"  M={partitions:>2} repeat {repeat_idx + 1}/{args.repeats}: "
                    f"observed_pieces={observed_piece_count:,} observed_change={observed_change:,} time={elapsed:.4f}s"
                )
            assert observed_piece_count is not None
            assert actual_blocks is not None
            timing = summarize_repeats(times)
            boundary_count = max(int(actual_blocks) - 1, 0)
            forced_extra = boundary_forced_extra(serial_flat, length, partitions)
            forced_piece_count = int(serial_piece_count + forced_extra)
            observed_change = int(observed_piece_count - serial_piece_count)
            rows.append(
                {
                    "length": int(length),
                    "kind": args.kind,
                    "seed": int(args.seed),
                    "tol": float(args.tol),
                    "partitions": int(partitions),
                    "n_jobs": int(partitions),
                    "actual_blocks": int(actual_blocks),
                    "serial_pieces": int(serial_piece_count),
                    "forced_parallel_pieces": int(forced_piece_count),
                    "forced_extra_pieces": int(forced_extra),
                    "observed_parallel_pieces": int(observed_piece_count),
                    "observed_piece_change": int(observed_change),
                    "theoretical_max_extra": boundary_count,
                    "forced_extra_ratio_percent": 100.0 * forced_extra / max(int(serial_piece_count), 1),
                    "forced_extra_per_boundary": float(forced_extra / boundary_count) if boundary_count else 0.0,
                    "observed_change_percent": 100.0 * observed_change / max(int(serial_piece_count), 1),
                    "compression_time_mean": timing["mean"],
                    "compression_time_std": timing["std"],
                    "compression_time_min": timing["min"],
                    "compression_time_max": timing["max"],
                    "compression_backend": jabba_compression.compression_backend(),
                    "warmup": int(args.warmup),
                    "repeats": int(args.repeats),
                }
            )

    csv_path = args.outdir / "compression_boundary_pieces.csv"
    tex_path = args.outdir / "compression_boundary_pieces.tex"
    write_csv(csv_path, rows)
    write_latex_table(tex_path, rows, args.lengths[0])
    plot_results(rows, args.outdir, args.formats, args.fontsize)

    print(f"\nCSV: {csv_path}")
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
