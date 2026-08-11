#!/usr/bin/env python3
"""Visualize MPI-JABBA UEA reconstructions against original time series.

Example:
mpirun -np 4 python visualize_uea_reconstruction.py \
  --dataset all \
  --data-dir ../UEA2018 \
  --outdir results_uea_total/figures \
  --formats pdf png

mpirun -np 4 python visualize_uea_reconstruction.py \
  --dataset all \
  --data-dir ../UEA2018 \
  --outdir results_uea_total/figures \
  --formats pdf png \
  --sample-index 0 \
  --dims 3
  
  """


from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from uea_total_runtime import (  # noqa: E402
    DATASETS,
    TOLS,
    digitize_with_revised_jabba_mpi,
    distributed_parallel_compress,
    inverse_compress_one,
    inverse_digitize_one,
    load_uea_dataset,
    make_model,
    mpi_context,
    normalize_rows,
    reshape_joint_dataset,
)


def set_plot_style(fontsize: int) -> None:
    sns.set_theme(context="paper", style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize + 1,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "figure.titlesize": fontsize + 2,
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.55,
            "lines.linewidth": 1.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "legend.handlelength": 1.8,
        }
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def trim_pair(original: np.ndarray, reconstructed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(original), len(reconstructed))
    return np.asarray(original[:n], dtype=np.float64), np.asarray(reconstructed[:n], dtype=np.float64)


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    original, reconstructed = trim_pair(original, reconstructed)
    if len(original) == 0:
        return math.nan
    diff = original - reconstructed
    return float(np.mean(diff * diff))


def symbol_colors(symbols_by_dim: list[list]) -> dict:
    unique = sorted({symbol for symbols in symbols_by_dim for symbol in list(symbols)}, key=str)
    palette = sns.color_palette("husl", n_colors=max(3, len(unique)))
    return {symbol: palette[i % len(palette)] for i, symbol in enumerate(unique)}


def symbol_segments(symbols, pieces: np.ndarray, target_length: int) -> list[tuple[float, float, object]]:
    if len(symbols) == 0 or len(pieces) == 0:
        return []
    lengths = np.maximum(np.rint(np.asarray(pieces[:, 0], dtype=np.float64)).astype(int), 1)
    target_sum = max(int(target_length) - 1, 1)
    lengths[-1] = max(1, int(lengths[-1]) + target_sum - int(lengths.sum()))
    starts = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(float)
    segments = []
    for start, width, symbol in zip(starts, lengths, symbols):
        if start >= target_sum:
            break
        clipped_width = max(0.5, min(float(width), float(target_sum) - float(start)))
        segments.append((float(start), clipped_width, symbol))
    return segments


def draw_symbol_ribbon(ax: plt.Axes, segments: list[tuple[float, float, object]], colors: dict) -> None:
    y0, y1 = ax.get_ylim()
    yrange = max(y1 - y0, 1e-9)
    ribbon_y = y0 + 0.045 * yrange
    ribbon_h = 0.052 * yrange
    for start, width, symbol in segments:
        ax.broken_barh(
            [(start, width)],
            (ribbon_y, ribbon_h),
            facecolors=colors.get(symbol, "#BBBBBB"),
            edgecolors="none",
            alpha=0.78,
            zorder=0,
        )


def reconstruct_selected_rows(model, symbols, row_indices: list[int], x_norm: np.ndarray):
    centers = np.asarray(model.parameters.centers, dtype=np.float64)
    alphabets = model.parameters.alphabets.tolist()
    starts = list(model.start_set or [])
    target_lengths = list(model.target_lengths_ or [len(row) for row in x_norm])

    reconstructions = []
    inverse_pieces = []
    for row_idx in row_indices:
        pieces = inverse_digitize_one(symbols[row_idx], centers, alphabets, int(target_lengths[row_idx]))
        reconstructed = inverse_compress_one(pieces, float(starts[row_idx]), int(target_lengths[row_idx]))
        reconstructions.append(np.asarray(reconstructed, dtype=np.float64))
        inverse_pieces.append(np.asarray(pieces, dtype=np.float64))
    return reconstructions, inverse_pieces


def plot_dataset_reconstruction(
    dataset: str,
    sample_index: int,
    dim_indices: list[int],
    x_norm: np.ndarray,
    original_shape: tuple[int, int, int],
    symbols,
    inverse_pieces: list[np.ndarray],
    reconstructions: list[np.ndarray],
    workers: int,
    outdir: Path,
    formats: list[str],
    fontsize: int,
) -> None:
    set_plot_style(fontsize)
    _, dim, length = original_shape
    row_indices = [sample_index * dim + dim_idx for dim_idx in dim_indices]
    colors = symbol_colors([list(symbols[row_idx]) for row_idx in row_indices])
    line_original = "#141414"
    line_reconstructed = "yellowgreen"

    fig_height = 1.72 * len(dim_indices) + 0.7
    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(7.3, fig_height), sharex=True)
    axes = np.atleast_1d(axes)
    x_axis = np.arange(length)

    for panel, (ax, dim_idx, row_idx, reconstructed, pieces) in enumerate(
        zip(axes, dim_indices, row_indices, reconstructions, inverse_pieces)
    ):
        original, reconstructed = trim_pair(x_norm[row_idx], reconstructed)
        x = x_axis[: len(original)]
        ax.plot(
            x,
            original,
            color=line_original,
            linewidth=1.65,
            linestyle=(0, (5, 2.2)),
            label="Original" if panel == 0 else None,
        )
        ax.plot(
            x,
            reconstructed,
            color=line_reconstructed,
            linewidth=1.85,
            alpha=0.94,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=line_reconstructed,
            markeredgewidth=1.0,
            markevery=max(1, len(x) // 32),
            label="MPI-JABBA reconstruction" if panel == 0 else None,
        )

        combined = np.concatenate([original, reconstructed]) if len(original) else np.array([0.0])
        y_min = float(np.nanmin(combined))
        y_max = float(np.nanmax(combined))
        pad = max((y_max - y_min) * 0.18, 0.35)
        ax.set_ylim(y_min - pad, y_max + pad * 0.68)
        draw_symbol_ribbon(ax, symbol_segments(symbols[row_idx], pieces, len(x_norm[row_idx])), colors)

        ax.set_ylabel(f"Dim {dim_idx + 1}")
        ax.text(
            0.985,
            0.84,
            f"MSE={mse(original, reconstructed):.3g}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=fontsize - 1,
            color="#333333",
        )
        ax.grid(True, alpha=0.30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Time index")
    axes[-1].set_xlim(0, max(length - 1, 1))
    fig.supylabel("Normalized value", x=0.015, fontsize=fontsize)
    fig.suptitle(f"{dataset}: MPI-JABBA reconstruction from symbolic sequences (M={workers})", y=0.995)

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#888888", edgecolor="none", alpha=0.78, label="Symbol sequence"))
    labels.append("Symbol sequence")
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.93))

    outdir.mkdir(parents=True, exist_ok=True)
    dim_label = f"dims{dim_indices[0] + 1}-{dim_indices[-1] + 1}"
    base = outdir / f"uea_reconstruction_{safe_name(dataset)}_sample{sample_index}_{dim_label}"
    for fmt in formats:
        fig.savefig(base.with_suffix(f".{fmt}"), dpi=400)
    plt.close(fig)


def visualize_dataset(name: str, args: argparse.Namespace) -> None:
    comm, rank, size, _, _ = mpi_context()
    tol = args.tol if args.tol is not None else TOLS[name]

    if rank == 0:
        multivariate_ts = load_uea_dataset(args.data_dir, name)
        sample_count, dim, _ = multivariate_ts.shape
        if args.sample_index < 0 or args.sample_index >= sample_count:
            raise IndexError(f"--sample-index {args.sample_index} is out of range for {name}; valid range: 0..{sample_count - 1}")
        x = reshape_joint_dataset(multivariate_ts)
        x_norm = normalize_rows(x).astype(np.float64)
        original_shape = tuple(int(v) for v in multivariate_ts.shape)
        dim_indices = list(range(min(args.dims, dim)))
        model = make_model(tol, args.alpha, args.sorting, args.scl, args.center_kind, prefer_mpi=True)
    else:
        x_norm = None
        original_shape = None
        dim_indices = None
        model = make_model(tol, args.alpha, args.sorting, args.scl, args.center_kind, prefer_mpi=True)

    pieces, _, _ = distributed_parallel_compress(
        model,
        x_norm,
        tol,
        args.alpha,
        args.sorting,
        args.scl,
        args.center_kind,
        args.compress_jobs,
    )
    symbols, _, _ = digitize_with_revised_jabba_mpi(model, x_norm, pieces)

    if rank == 0:
        assert x_norm is not None
        assert symbols is not None
        assert original_shape is not None
        assert dim_indices is not None
        sample_count, dim, _ = original_shape
        row_indices = [args.sample_index * dim + dim_idx for dim_idx in dim_indices]
        reconstructions, inverse_pieces = reconstruct_selected_rows(model, symbols, row_indices, x_norm)
        plot_dataset_reconstruction(
            name,
            args.sample_index,
            dim_indices,
            x_norm,
            original_shape,
            symbols,
            inverse_pieces,
            reconstructions,
            size,
            args.outdir,
            args.formats,
            args.fontsize,
        )
        print(f"Saved {name} reconstruction figure(s) to {args.outdir}", flush=True)

    comm.Barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UEA MPI-JABBA symbolic reconstructions.")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument("--data-dir", type=Path, default=Path("../UEA2018"))
    parser.add_argument("--outdir", type=Path, default=Path("results_uea_total/figures"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--dims", type=int, default=3, help="Number of leading dimensions to visualize.")
    parser.add_argument("--tol", type=float, default=None, help="Override the dataset-specific compression tolerance.")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--compress-jobs", type=int, default=0)
    parser.add_argument("--sorting", default="2-norm", choices=("2-norm", "1-norm", "lexi"))
    parser.add_argument("--scl", type=float, default=1.0)
    parser.add_argument("--center-kind", default="seed", choices=("seed", "centroid"))
    parser.add_argument("--fontsize", type=int, default=10)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    args.outdir = args.outdir.resolve() if args.outdir.is_absolute() else (Path.cwd() / args.outdir).resolve()
    args.data_dir = args.data_dir.resolve() if args.data_dir.is_absolute() else (Path.cwd() / args.data_dir).resolve()
    if args.dims <= 0:
        raise ValueError("--dims must be positive")

    comm, rank, size, _, _ = mpi_context()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    unknown = [dataset for dataset in datasets if dataset not in TOLS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Known datasets: {list(DATASETS)}")

    if rank == 0:
        args.outdir.mkdir(parents=True, exist_ok=True)
        print(f"UEA reconstruction visualization: M={size}, datasets={list(datasets)}, output={args.outdir}", flush=True)

    for dataset in datasets:
        visualize_dataset(dataset, args)


if __name__ == "__main__":
    main()
