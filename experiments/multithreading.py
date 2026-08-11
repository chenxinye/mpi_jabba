"""
Multithreading utilities for time series experiments.

python multithreading.py \
  --lengths 100000 \
  --threads 2 3 4 5 6 7 8 \
  --tol 0.01 \
  --alpha 0.05 \
  --kind gaussian \
  --warmup 1 \
  --repeats 3 \
  --outdir results_univariate_scaling \
  --formats pdf png
  
"""


from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

from ABBA import ABBA
from fABBA import fABBA
from common import gather_global_labels, import_mpi_alphaagg, sorted_blocks
from jabba import JABBA, Model, symbolsAssign
from jabba.aggregation import labels_centers_from_output
from jabba.jabba import one_D_centers
import jabba.compression as jabba_compression


# warnings.simplefilter(action="ignore", category=FutureWarning)


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def default_threads(max_threads: int) -> list[int]:
    candidates = [1, 2, 4, 8, 16, 32, 64]
    return [t for t in candidates if t <= max_threads]


def running_under_mpi() -> bool:
    keys = (
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "PMI_SIZE",
        "PMI_RANK",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
    )
    return any(key in os.environ for key in keys)


def mpi_context():
    if not running_under_mpi():
        return None, 0, 1, time.perf_counter, None
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size(), MPI.Wtime, MPI.MAX


def maybe_spawn_mpi(args: argparse.Namespace) -> None:
    if running_under_mpi() or os.environ.get("JABBA_MULTITHREADING_MPI_CHILD") == "1":
        return
    if args.mpi_processes <= 1:
        return
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        raise RuntimeError("mpirun was not found; use --mpi-processes 1 or install/configure MPI.")
    env = os.environ.copy()
    env["JABBA_MULTITHREADING_MPI_CHILD"] = "1"
    cmd = [mpirun, "-np", str(args.mpi_processes), sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.run(cmd, env=env, check=False).returncode)


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
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.0,
        }
    )


def plot_ci_curve(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    color: str,
    label: str | None = None,
    order: int = 1,
    marker: str = "o",
    scatter_size: float = 35.0,
    fit_x: str = "linear",
    fit_y: str = "linear",
    ci: float = 95.0,
) -> None:
    """Draw a notebook-style fitted curve with a deterministic confidence band."""
    xy = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    xy = xy.replace([np.inf, -np.inf], np.nan).dropna()
    if fit_x == "log2":
        xy = xy[xy["x"] > 0]
    if fit_y == "log":
        xy = xy[xy["y"] > 0]
    if xy.empty:
        return

    x_vals = xy["x"].to_numpy(dtype=float)
    y_vals = xy["y"].to_numpy(dtype=float)
    x_fit = np.log2(x_vals) if fit_x == "log2" else x_vals
    y_fit = np.log(y_vals) if fit_y == "log" else y_vals
    degree = min(order, len(xy) - 1)

    if len(xy) >= max(degree + 2, 3) and degree >= 1:
        if fit_x == "log2":
            grid = np.geomspace(float(x_vals.min()), float(x_vals.max()), 200)
            grid_fit = np.log2(grid)
        else:
            grid = np.linspace(float(x_vals.min()), float(x_vals.max()), 200)
            grid_fit = grid

        coeffs = np.polyfit(x_fit, y_fit, degree)
        fitted = np.polyval(coeffs, grid_fit)

        rng = np.random.default_rng(0)
        boot = []
        for _ in range(1000):
            sample = rng.integers(0, len(x_fit), len(x_fit))
            if len(np.unique(x_fit[sample])) <= degree:
                continue
            boot_coeffs = np.polyfit(x_fit[sample], y_fit[sample], degree)
            boot.append(np.polyval(boot_coeffs, grid_fit))

        if fit_y == "log":
            fitted = np.exp(fitted)
            boot = [np.exp(vals) for vals in boot]

        ax.plot(grid, fitted, color=color, linewidth=2.15, label=label, zorder=2)
        if boot:
            alpha = (100.0 - ci) / 2.0
            lower, upper = np.percentile(np.vstack(boot), [alpha, 100.0 - alpha], axis=0)
            if fit_y == "linear":
                lower = np.maximum(lower, 0.0)
            ax.fill_between(grid, lower, upper, color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.scatter(
            x_vals,
            y_vals,
            s=scatter_size,
            facecolors="white",
            edgecolors=color,
            linewidths=1.35,
            marker=marker,
            zorder=3,
        )
        return

    ax.plot(
        xy["x"],
        xy["y"],
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.35,
        color=color,
        label=label,
    )


def add_errorbar_caps(ax: plt.Axes, x: pd.Series, y: pd.Series, yerr: pd.Series, color: str) -> None:
    err = pd.to_numeric(yerr, errors="coerce")
    if not np.isfinite(err).any() or float(err.fillna(0).abs().max()) == 0.0:
        return
    ax.errorbar(
        x,
        y,
        yerr=err,
        fmt="none",
        ecolor=color,
        elinewidth=0.85,
        capsize=2.6,
        alpha=0.55,
        zorder=1,
    )


def make_series(length: int, seed: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.normal(0.0, 1.0, length).astype(np.float64)
    if kind == "random_walk":
        return np.cumsum(rng.normal(0.0, 1.0, length)).astype(np.float64)
    if kind == "seasonal":
        x = np.linspace(0.0, 80.0 * np.pi, length)
        trend = 0.15 * np.sin(x / 30.0)
        signal = np.sin(x) + 0.5 * np.sin(0.2 * x + 0.5) + trend
        return (signal + 0.35 * rng.normal(size=length)).astype(np.float64)
    raise ValueError("kind must be one of: gaussian, random_walk, seasonal")


def mse_trimmed(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(mean_squared_error(a[:n], b[:n]))


def flatten_symbols(symbols) -> np.ndarray:
    if isinstance(symbols, (str, tuple)):
        return np.asarray(list(symbols))
    if isinstance(symbols, list) and symbols and isinstance(symbols[0], (list, tuple, np.ndarray)):
        return np.asarray([item for seq in symbols for item in seq])
    return np.asarray(symbols)


def count_symbols(symbols) -> int:
    flat = flatten_symbols(symbols)
    return int(len(np.unique(flat)))


def timed_runs(fn, warmup: int, repeats: int) -> tuple[list[dict], dict]:
    for _ in range(warmup):
        fn()

    runs = []
    for repeat in range(repeats):
        result = fn()
        if result is not None:
            result["repeat"] = repeat
            runs.append(result)

    if not runs:
        return [], {}

    numeric_keys = [k for k, v in runs[0].items() if isinstance(v, (int, float, np.integer, np.floating))]
    summary = {}
    for key in numeric_keys:
        vals = np.asarray([row[key] for row in runs], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals, ddof=0))
        summary[f"{key}_min"] = float(np.min(vals))
        summary[f"{key}_max"] = float(np.max(vals))
    return runs, summary


def run_fabba(series: np.ndarray, tol: float, alpha: float) -> dict:
    start = time.perf_counter()
    model = fABBA(tol=tol, alpha=alpha, sorting="norm", scl=1, verbose=0, return_list=True)
    symbols = model.fit_transform(series)
    reconstruction = model.inverse_transform(symbols, series[0])
    elapsed = time.perf_counter() - start
    return {
        "runtime": elapsed,
        "mse": mse_trimmed(series, reconstruction),
        "symbols": count_symbols(symbols),
        "centers": int(model.parameters.centers.shape[0]),
        "pieces": int(len(symbols)),
        "boundary_extra": 0,
        "compression_time": np.nan,
        "digitization_time": np.nan,
        "inverse_time": np.nan,
        "reconstruction": np.asarray(reconstruction, dtype=np.float64),
    }


def run_abba(series: np.ndarray, tol: float, symbols_k: int) -> dict:
    start = time.perf_counter()
    model = ABBA(tol=tol, scl=1, min_k=symbols_k, max_k=symbols_k, norm=2, verbose=0)
    symbols, centers = model.transform(series)
    reconstruction = model.inverse_transform(symbols, np.float32(centers), series[0])
    elapsed = time.perf_counter() - start
    return {
        "runtime": elapsed,
        "mse": mse_trimmed(series, reconstruction),
        "symbols": count_symbols(symbols),
        "centers": int(centers.shape[0]),
        "pieces": int(len(symbols)),
        "boundary_extra": 0,
        "compression_time": np.nan,
        "digitization_time": np.nan,
        "inverse_time": np.nan,
        "reconstruction": np.asarray(reconstruction, dtype=np.float64),
    }


def legacy_inv_digitize(strings, centers: np.ndarray, alphabets: list) -> np.ndarray:
    return np.vstack([centers[alphabets.index(symbol)][:2] for symbol in strings])


def legacy_quantize(pieces: np.ndarray) -> np.ndarray:
    pieces = np.asarray(pieces, dtype=np.float64).copy()
    if len(pieces) == 0:
        return pieces
    if len(pieces) == 1:
        pieces[0, 0] = round(pieces[0, 0])
        return pieces
    for idx in range(len(pieces) - 1):
        corr = round(pieces[idx, 0]) - pieces[idx, 0]
        pieces[idx, 0] = round(pieces[idx, 0] + corr)
        pieces[idx + 1, 0] = pieces[idx + 1, 0] - corr
        if pieces[idx, 0] == 0:
            pieces[idx, 0] = 1
            pieces[idx + 1, 0] -= 1
    pieces[-1, 0] = round(pieces[-1, 0], 0)
    return pieces


def legacy_inv_compress(pieces: np.ndarray, start: float) -> list[float]:
    time_series = [float(start)]
    for idx in range(len(pieces)):
        length = pieces[idx, 0]
        inc = pieces[idx, 1]
        x = np.arange(0, length + 1) / length * inc
        y = time_series[-1] + x
        time_series = time_series + y[1:].tolist()
    return time_series


def legacy_inv_transform(strings, centers: np.ndarray, alphabets: list, start: float) -> list[float]:
    pieces = legacy_inv_digitize(strings, centers, alphabets)
    pieces = legacy_quantize(pieces)
    return legacy_inv_compress(pieces, start)


def legacy_inverse_transform(model: JABBA, string_sequences, n_jobs: int) -> np.ndarray:
    if model.parameters is None:
        raise ValueError("Please fit the model before inverse_transform.")
    if model.start_set is None:
        raise ValueError("Please input valid start_set.")
    n_jobs = model.n_jobs_init(n_jobs)
    if isinstance(string_sequences, (str, tuple)):
        string_sequences = [list(string_sequences)]

    centers = np.asarray(model.parameters.centers, dtype=np.float64)
    alphabets = model.parameters.alphabets.tolist()
    count = len(string_sequences)
    if n_jobs != 1 and count != 1:
        pool = Pool(n_jobs)
        jobs = [
            pool.apply_async(
                legacy_inv_transform,
                args=(string_sequences[i], centers, alphabets, model.start_set[i]),
            )
            for i in range(count)
        ]
        pool.close()
        pool.join()
        inverse_sequences = [job.get() for job in jobs]
    else:
        inverse_sequences = [
            legacy_inv_transform(seq, centers, alphabets, model.start_set[i])
            for i, seq in enumerate(string_sequences)
        ]
    if model.return_series_univariate:
        return np.hstack(inverse_sequences)
    return np.asarray(inverse_sequences, dtype=object)


def _normalize_sorting(sorting: str) -> str:
    return "2-norm" if sorting == "norm" else sorting


def mpi_ptga_labels_and_centers(points: np.ndarray | None, alpha: float, sorting: str, center_kind: str):
    comm, rank, size, _, _ = mpi_context()
    if size == 1:
        from jabba.aggregation import aggregate_points

        assert points is not None
        out = aggregate_points(points, alpha, sorting, algorithm="serial", prefer_mpi=False)
        labels, centers_scaled = labels_centers_from_output(out, center_kind=center_kind)
        return labels, np.ascontiguousarray(centers_scaled, dtype=np.float64), out

    _, aggregate_mpi_ptga = import_mpi_alphaagg()
    sorting = _normalize_sorting(sorting)
    if rank == 0:
        assert points is not None
        n_points = int(points.shape[0])
        data_blocks, index_blocks = sorted_blocks(points, size, sorting)
    else:
        n_points = None
        data_blocks = None
        index_blocks = None

    n_points = comm.bcast(n_points, root=0)
    local_points = comm.scatter(data_blocks, root=0)
    local_indices = comm.scatter(index_blocks, root=0)

    out = aggregate_mpi_ptga(local_points, alpha, sorting)
    local_labels = np.asarray(out["labels"], dtype=np.int64)
    labels = gather_global_labels(comm, local_indices, local_labels, n_points)
    if rank != 0:
        return None, None, None

    out = dict(out)
    out["labels"] = labels
    labels, centers_scaled = labels_centers_from_output(out, center_kind=center_kind)
    return labels, np.ascontiguousarray(centers_scaled, dtype=np.float64), out


def digitize_jabba_mpi(model: JABBA, series: np.ndarray | None, pieces, alphabet_set: int = 0):
    comm, rank, _, wall_time, reduce_max = mpi_context()
    if comm is not None:
        comm.Barrier()
    t0 = wall_time()

    if rank == 0:
        assert series is not None
        assert pieces is not None
        series = np.asarray(series)
        len_ts = len(series)
        model.eta = 0.000002 if series.ndim > 1 else 0.01
        num_pieces = [len(piece) for piece in pieces]
        flat = np.vstack(pieces)[:, :2]
        model._std = np.std(flat, axis=0)
        model._std[model._std == 0] = 1
        len_pieces = flat[:, 0].copy()
        scaled = flat * np.array([model.scl, 1.0]) / model._std
        max_k = np.unique(scaled[:, :2], axis=0).shape[0]
        if model.auto_digitize:
            sum_of_length = sum(len(series[i]) for i in range(len_ts)) if series.ndim > 1 else len_ts
            denom = max(max_k * (model.eta**2) * (3 * (sum_of_length**4) + 2 - 5 * (max_k**2)), np.finfo(float).eps)
            numer = 60 * sum_of_length * max(sum_of_length - max_k, 1) * (model.tol**2)
            model.alpha = pow(numer / denom, 1 / 4)
        alpha = float(model.alpha)
        sorting = model.sorting
    else:
        num_pieces = None
        len_pieces = None
        scaled = None
        alpha = None
        sorting = None

    if comm is not None:
        alpha = comm.bcast(alpha, root=0)
        sorting = comm.bcast(sorting, root=0)
    labels, centers_scaled, out = mpi_ptga_labels_and_centers(scaled, alpha, sorting, model.center_kind)

    if rank == 0:
        assert labels is not None
        assert centers_scaled is not None
        centers = centers_scaled * model._std / np.array([model.scl, 1.0])
        model.k = centers.shape[0]
        if model.scl == 0:
            centers[:, 0] = one_D_centers(len_pieces, labels, model.k)
        string, alphabets = symbolsAssign(labels, alphabet_set)
        model.parameters = Model(np.asarray(centers, dtype=np.float64), alphabets)
        model.num_grp = model.parameters.centers.shape[0]
        model.aggregation_backend_ = str(out.get("backend", "mpi-alpha-agg:aggregate_mpi_ptga")) if out else "unknown"
        symbols = model.string_separation(string, num_pieces)
    else:
        symbols = None

    if comm is not None:
        comm.Barrier()
    elapsed = wall_time() - t0
    if comm is not None and reduce_max is not None:
        elapsed = comm.allreduce(elapsed, op=reduce_max)
    return symbols, float(elapsed)


def split_series_for_mpi(series: np.ndarray, workers: int) -> list[np.ndarray]:
    blocks = np.array_split(np.asarray(series, dtype=np.float64), workers)
    return [np.ascontiguousarray(block, dtype=np.float64) for block in blocks]


def compress_local_block(
    block: np.ndarray,
    tol: float,
    alpha: float,
    max_len: float,
    fillna_method: str,
    n_jobs: int,
    compression_backend: str,
):
    jabba_compression.set_compression_backend(compression_backend)
    if len(block) < 2:
        return [], [], []

    local_model = JABBA(
        tol=tol,
        init="agg",
        alpha=alpha,
        sorting="norm",
        auto_digitize=False,
        partition=max(int(n_jobs), 1),
        verbose=0,
        prefer_mpi=False,
        center_kind="seed",
        max_len=max_len,
        fillna=fillna_method,
    )
    pieces = local_model.parallel_compress(block, n_jobs=max(int(n_jobs), 1))
    return pieces, local_model.start_set, local_model.target_lengths_


def mpi_parallel_compress(series: np.ndarray | None, model: JABBA, threads: int, compression_backend: str):
    comm, rank, size, wall_time, reduce_max = mpi_context()
    if comm is None or size <= 1:
        if rank == 0:
            assert series is not None
            jabba_compression.set_compression_backend(compression_backend)
            t0 = time.perf_counter()
            pieces = model.parallel_compress(series, n_jobs=threads)
            elapsed = time.perf_counter() - t0
            return pieces, elapsed
        return None, 0.0

    if rank == 0:
        assert series is not None
        blocks = split_series_for_mpi(series, size)
    else:
        blocks = None

    local_block = comm.scatter(blocks, root=0)
    local_jobs = max(1, int(np.ceil(max(threads, 1) / size)))

    comm.Barrier()
    t0 = wall_time()
    local_pieces, local_starts, local_lengths = compress_local_block(
        local_block,
        model.tol,
        model.alpha,
        model.max_len,
        model.fillna,
        local_jobs,
        compression_backend,
    )
    comm.Barrier()
    elapsed = wall_time() - t0
    elapsed = comm.allreduce(elapsed, op=reduce_max)

    gathered_pieces = comm.gather(local_pieces, root=0)
    gathered_starts = comm.gather(local_starts, root=0)
    gathered_lengths = comm.gather(local_lengths, root=0)

    if rank != 0:
        return None, float(elapsed)

    pieces = [np.asarray(piece, dtype=np.float64) for rank_pieces in gathered_pieces for piece in rank_pieces]
    model.start_set = [float(start) for rank_starts in gathered_starts for start in rank_starts]
    model.target_lengths_ = [int(length) for rank_lengths in gathered_lengths for length in rank_lengths]
    model.return_series_univariate = True
    model.compression_backend_ = jabba_compression.compression_backend()
    return pieces, float(elapsed)


def run_jabba(
    series: np.ndarray | None,
    tol: float,
    alpha: float,
    threads: int,
    compression_backend: str,
    inverse_backend: str,
) -> dict | None:
    comm, rank, size, _, _ = mpi_context()
    original_backend = jabba_compression._compression_backend_choice
    jabba_compression.set_compression_backend(compression_backend)

    model = JABBA(
        tol=tol,
        alpha=alpha,
        init="agg",
        sorting="norm",
        auto_digitize=False,
        partition=threads,
        verbose=0,
        prefer_mpi=False,
        center_kind="seed",
    )

    try:
        pieces, compression_time = mpi_parallel_compress(series, model, threads, compression_backend)

        if size > 1:
            symbols, digitization_time = digitize_jabba_mpi(model, series if rank == 0 else None, pieces)
            t2 = time.perf_counter() if rank == 0 else None
        else:
            t_digit0 = time.perf_counter()
            symbols = model.digitize(series, pieces, n_jobs=threads)
            t2 = time.perf_counter()
            digitization_time = t2 - t_digit0

        if rank != 0:
            return None

        if inverse_backend == "legacy-python":
            reconstruction = legacy_inverse_transform(model, symbols, n_jobs=threads)
        else:
            reconstruction = model.inverse_transform(symbols, n_jobs=threads)
        t3 = time.perf_counter()
        inverse_time = t3 - t2
    finally:
        jabba_compression.set_compression_backend(original_backend)

    total_pieces = int(sum(len(piece) for piece in pieces))
    return {
        "runtime": float(compression_time + digitization_time + inverse_time),
        "fit_runtime": float(compression_time + digitization_time),
        "mse": mse_trimmed(series, reconstruction),
        "symbols": count_symbols(symbols),
        "centers": int(model.parameters.centers.shape[0]),
        "pieces": total_pieces,
        "boundary_extra": int(max(0, total_pieces - model.parameters.centers.shape[0])),
        "compression_time": compression_time,
        "digitization_time": digitization_time,
        "inverse_time": inverse_time,
        "mpi_processes": size,
        "aggregation_backend": model.aggregation_backend_,
        "compression_backend": model.compression_backend_,
        "requested_compression_backend": compression_backend,
        "inverse_backend": inverse_backend,
        "reconstruction": np.asarray(reconstruction, dtype=np.float64),
    }


def summarize_method(method: str, length: int, threads: int | None, runs: list[dict], summary: dict) -> dict:
    row = {
        "method": method,
        "length": length,
        "threads": threads if threads is not None else 1,
        "partitions": threads if method == "JABBA" else 1,
    }
    row.update(summary)
    last = runs[-1]
    row["symbols_last"] = int(last["symbols"])
    row["centers_last"] = int(last["centers"])
    row["pieces_last"] = int(last["pieces"])
    row["compression_backend"] = last.get("compression_backend", "")
    row["requested_compression_backend"] = last.get("requested_compression_backend", "")
    row["aggregation_backend"] = last.get("aggregation_backend", "")
    row["inverse_backend"] = last.get("inverse_backend", "")
    return row


def save_records(rows: list[dict], path: Path, speedup_metric: str = "fit") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    jabba_one = df[(df["method"] == "JABBA") & (df["threads"] == 1)].copy()
    fit_base = dict(zip(jabba_one["length"], jabba_one.get("fit_runtime_mean", jabba_one["runtime_mean"])))
    total_base = dict(zip(jabba_one["length"], jabba_one["runtime_mean"]))
    compression_base = dict(zip(jabba_one["length"], jabba_one.get("compression_time_mean", np.nan)))
    df["jabba_fit_speedup"] = [
        fit_base.get(row.length, np.nan) / row.fit_runtime_mean if row.method == "JABBA" and getattr(row, "fit_runtime_mean", 0) > 0 else np.nan
        for row in df.itertuples()
    ]
    df["jabba_total_speedup"] = [
        total_base.get(row.length, np.nan) / row.runtime_mean if row.method == "JABBA" and row.runtime_mean > 0 else np.nan
        for row in df.itertuples()
    ]
    df["jabba_compression_speedup"] = [
        compression_base.get(row.length, np.nan) / row.compression_time_mean
        if row.method == "JABBA" and getattr(row, "compression_time_mean", 0) > 0
        else np.nan
        for row in df.itertuples()
    ]
    metric_columns = {
        "fit": "jabba_fit_speedup",
        "total": "jabba_total_speedup",
        "compression": "jabba_compression_speedup",
    }
    df["jabba_speedup_metric"] = speedup_metric
    df["jabba_speedup"] = df[metric_columns.get(speedup_metric, "jabba_fit_speedup")]
    df["jabba_efficiency"] = [
        row.jabba_speedup / row.threads if row.method == "JABBA" else np.nan
        for row in df.itertuples()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def plot_overview(df: pd.DataFrame, outdir: Path, fontsize: int, formats: list[str]) -> None:
    set_plot_style(fontsize)
    length = sorted(df["length"].unique())[-1]
    sub = df[df["length"] == length].copy()
    jabba = sub[sub["method"] == "JABBA"].sort_values("threads")
    abba_rows = sub[sub["method"] == "ABBA"]
    fabba_rows = sub[sub["method"] == "fABBA"]
    abba = abba_rows.iloc[0] if len(abba_rows) else None
    fabba = fabba_rows.iloc[0] if len(fabba_rows) else None

    palette = {"JABBA": "#7B3294", "ABBA": "#D55E00", "fABBA": "#009E73", "ideal": "#333333"}

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.15))
    ax = axes[0, 0]
    plot_ci_curve(
        ax,
        jabba["threads"],
        jabba["runtime_mean"],
        palette["JABBA"],
        label="JABBA",
        order=1,
        fit_x="log2",
        fit_y="log",
    )
    add_errorbar_caps(ax, jabba["threads"], jabba["runtime_mean"], jabba["runtime_std"], palette["JABBA"])
    if abba is not None:
        ax.axhline(abba["runtime_mean"], color=palette["ABBA"], linestyle=(0, (5, 2.5)), linewidth=1.65, label="ABBA")
    if fabba is not None:
        ax.axhline(fabba["runtime_mean"], color=palette["fABBA"], linestyle=(0, (3, 2, 1, 2)), linewidth=1.65, label="fABBA")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(jabba["threads"])
    ax.set_xticklabels([str(int(t)) for t in jabba["threads"]])
    ax.set_xlabel("Threads / partitions")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime, n={length:,}")
    ax.legend(frameon=False, loc="best")

    ax = axes[0, 1]
    speed = jabba.dropna(subset=["jabba_speedup"])
    if len(speed):
        plot_ci_curve(
            ax,
            speed["threads"],
            speed["jabba_speedup"],
            palette["JABBA"],
            label="Measured",
            order=2,
            fit_x="log2",
        )
        ax.plot(speed["threads"], speed["threads"], linestyle=(0, (1.2, 2.0)), color=palette["ideal"], linewidth=1.4, label="Ideal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(speed["threads"])
        ax.set_xticklabels([str(int(t)) for t in speed["threads"]])
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Requires JABBA-1 baseline", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Threads / partitions")
    ax.set_ylabel("Speedup vs JABBA-1")
    metric = "fit"
    if "jabba_speedup_metric" in jabba.columns and len(jabba["jabba_speedup_metric"].dropna()):
        metric = str(jabba["jabba_speedup_metric"].dropna().iloc[0])
    title_map = {
        "fit": "Fit-phase speedup",
        "total": "Total-runtime speedup",
        "compression": "Compression speedup",
    }
    ax.set_title(title_map.get(metric, "JABBA speedup"))

    ax = axes[1, 0]
    efficiency = jabba.dropna(subset=["jabba_efficiency"])
    if len(efficiency):
        plot_ci_curve(
            ax,
            efficiency["threads"],
            efficiency["jabba_efficiency"],
            palette["JABBA"],
            order=2,
            fit_x="log2",
        )
        ax.axhline(1.0, color=palette["ideal"], linestyle=(0, (1.2, 2.0)), linewidth=1.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(efficiency["threads"])
        ax.set_xticklabels([str(int(t)) for t in efficiency["threads"]])
        ax.set_ylim(0, max(1.05, float(efficiency["jabba_efficiency"].max()) * 1.08))
    else:
        ax.text(0.5, 0.5, "Requires JABBA-1 baseline", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Threads / partitions")
    ax.set_ylabel("Efficiency")
    ax.set_title("Efficiency")

    ax = axes[1, 1]
    plot_ci_curve(
        ax,
        jabba["threads"],
        jabba["mse_mean"],
        palette["JABBA"],
        label="JABBA",
        order=1,
        fit_x="log2",
        fit_y="log",
    )
    if abba is not None:
        ax.axhline(abba["mse_mean"], color=palette["ABBA"], linestyle=(0, (5, 2.5)), linewidth=1.65, label="ABBA")
    if fabba is not None:
        ax.axhline(fabba["mse_mean"], color=palette["fABBA"], linestyle=(0, (3, 2, 1, 2)), linewidth=1.65, label="fABBA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(jabba["threads"])
    ax.set_xticklabels([str(int(t)) for t in jabba["threads"]])
    ax.set_xlabel("Threads / partitions")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction quality")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(True, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"univariate_scaling_overview.{fmt}", dpi=400)
    plt.close(fig)


def plot_breakdown(df: pd.DataFrame, outdir: Path, fontsize: int, formats: list[str]) -> None:
    set_plot_style(fontsize)
    length = sorted(df["length"].unique())[-1]
    jabba = df[(df["length"] == length) & (df["method"] == "JABBA")].sort_values("threads")
    phases = ["compression_time_mean", "digitization_time_mean", "inverse_time_mean"]
    labels = ["Compression", "Digitization", "Inverse"]
    colors = ["#0072B2", "#CC79A7", "#E69F00"]

    fig, ax = plt.subplots(figsize=(6.9, 3.1))
    bottom = np.zeros(len(jabba))
    x = np.arange(len(jabba))
    for phase, label, color in zip(phases, labels, colors):
        vals = jabba[phase].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.72)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in jabba["threads"]])
    ax.set_xlabel("Threads / partitions")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"JABBA runtime decomposition, n={length:,}")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(outdir / f"univariate_jabba_breakdown.{fmt}", dpi=400)
    plt.close(fig)


def plot_reconstruction(series: np.ndarray, examples: dict[str, np.ndarray], outdir: Path, fontsize: int, formats: list[str], sample_points: int) -> None:
    set_plot_style(fontsize)
    n = min(sample_points, len(series))
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.plot(x, series[:n], color="#111111", linewidth=1.8, label="Original")
    styles = {"ABBA": ("#D55E00", "--"), "fABBA": ("#009E73", "-."), "JABBA": ("#7B3294", "-")}
    for name, rec in examples.items():
        color, linestyle = styles.get(name, ("#666666", "-"))
        ax.plot(x, rec[:n], color=color, linestyle=linestyle, linewidth=1.6, label=name)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Value")
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.26))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.25)
    for fmt in formats:
        fig.savefig(outdir / f"univariate_reconstruction_example.{fmt}", dpi=400)
    plt.close(fig)


def write_summary(df: pd.DataFrame, path: Path) -> None:
    summary = {"lengths": sorted(map(int, df["length"].unique()))}
    for length in summary["lengths"]:
        sub = df[df["length"] == length]
        jabba = sub[sub["method"] == "JABBA"].sort_values("threads")
        best = jabba.loc[jabba["runtime_mean"].idxmin()]
        item = {
            "best_jabba_threads": int(best["threads"]),
            "best_jabba_runtime": float(best["runtime_mean"]),
            "best_jabba_speedup": float(best["jabba_speedup"]),
            "best_jabba_efficiency": float(best["jabba_efficiency"]),
        }
        fbase_rows = sub[sub["method"] == "fABBA"]
        if len(fbase_rows):
            item["fabba_runtime"] = float(fbase_rows["runtime_mean"].iloc[0])
        abba_rows = sub[sub["method"] == "ABBA"]
        if len(abba_rows):
            item["abba_runtime"] = float(abba_rows["runtime_mean"].iloc[0])
        summary[str(length)] = item
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def resolve_threads(args: argparse.Namespace, max_cpu: int) -> list[int]:
    threads = args.threads if args.threads is not None else default_threads(max_cpu)
    threads = sorted(set(int(t) for t in threads if int(t) > 0))
    if not args.allow_oversubscribe:
        threads = [t for t in threads if t <= max_cpu]
    if not threads:
        raise SystemExit("No valid --threads values remain after CPU filtering.")
    return threads


def driver_argv(args: argparse.Namespace, threads: int, outdir: Path, skip_baselines: bool = False) -> list[str]:
    argv = [
        "--lengths",
        *[str(v) for v in args.lengths],
        "--threads",
        str(threads),
        "--tol",
        str(args.tol),
        "--alpha",
        str(args.alpha),
        "--seed",
        str(args.seed),
        "--kind",
        args.kind,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--baseline-warmup",
        str(args.baseline_warmup),
        "--baseline-repeats",
        str(args.baseline_repeats),
        "--outdir",
        str(outdir),
        "--fontsize",
        str(args.fontsize),
        "--formats",
        *args.formats,
        "--sample-points",
        str(args.sample_points),
        "--mpi-processes",
        str(threads),
        "--jabba-compression-backend",
        args.jabba_compression_backend,
        "--jabba-inverse-backend",
        args.jabba_inverse_backend,
        "--jabba-speedup-metric",
        args.jabba_speedup_metric,
        "--skip-plots",
    ]
    if args.allow_oversubscribe:
        argv.append("--allow-oversubscribe")
    if args.skip_abba:
        argv.append("--skip-abba")
    if skip_baselines:
        argv.append("--skip-baselines")
    return argv


def run_same_mpi_driver(args: argparse.Namespace, threads: list[int], max_cpu: int) -> None:
    if running_under_mpi() or args.mpi_processes != 0:
        return
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        raise RuntimeError("mpirun was not found; specify --mpi-processes 1 or install/configure MPI.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    run_root = Path(tempfile.mkdtemp(prefix="same_mpi_", dir=args.outdir))
    env = os.environ.copy()
    env["JABBA_MULTITHREADING_MPI_CHILD"] = "1"
    run_threads = list(threads)
    if 1 not in run_threads:
        run_threads = [1] + run_threads

    print("Auto MPI mode: using M = threads for each JABBA run.")
    print(f"Requested threads/M values: {threads}")
    if run_threads != threads:
        print("Adding M=1 JABBA baseline for speedup calculation.")
    print(f"Executed threads/M values: {run_threads}")
    print(f"Temporary per-M outputs: {run_root}")

    csv_paths = []
    for idx, t in enumerate(run_threads):
        subdir = run_root / f"M{t}"
        subdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            mpirun,
            "-np",
            str(t),
            sys.executable,
            str(Path(__file__).resolve()),
            *driver_argv(args, t, subdir, skip_baselines=(idx > 0)),
        ]
        print(f"\n=== launching JABBA multithreading experiment with threads=M={t} ===", flush=True)
        completed = subprocess.run(cmd, env=env, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        csv_path = subdir / "univariate_thread_scaling.csv"
        if not csv_path.exists():
            raise SystemExit(f"Expected CSV was not produced: {csv_path}")
        csv_paths.append(csv_path)

    combined = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    baselines = combined[combined["method"] != "JABBA"].drop_duplicates(["method", "length"], keep="first")
    jabba = combined[combined["method"] == "JABBA"].drop_duplicates(["method", "length", "threads"], keep="last")
    rows_df = pd.concat([baselines, jabba], ignore_index=True).sort_values(["length", "method", "threads"])

    df = save_records(rows_df.to_dict("records"), args.outdir / "univariate_thread_scaling.csv", args.jabba_speedup_metric)
    write_summary(df, args.outdir / "univariate_thread_scaling_summary.json")
    plot_overview(df, args.outdir, args.fontsize, args.formats)
    plot_breakdown(df, args.outdir, args.fontsize, args.formats)

    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": max_cpu,
        "threads_requested": threads,
        "threads": run_threads,
        "mpi_processes_requested": "same-as-threads",
        "mpi_processes_actual": "per-run",
        "jabba_compression_backend": args.jabba_compression_backend,
        "jabba_inverse_backend": args.jabba_inverse_backend,
        "jabba_speedup_metric": args.jabba_speedup_metric,
        "per_run_output_root": str(run_root),
    }
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nSaved combined results to {args.outdir}")
    print(f"CSV: {args.outdir / 'univariate_thread_scaling.csv'}")
    for fmt in args.formats:
        print(f"Overview figure: {args.outdir / f'univariate_scaling_overview.{fmt}'}")
    raise SystemExit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Univariate ABBA/fABBA/JABBA thread scaling experiment.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[100000], help="Time-series lengths.")
    parser.add_argument("--threads", type=int, nargs="+", default=None, help="JABBA threads/partitions to test.")
    parser.add_argument("--allow-oversubscribe", action="store_true", help="Keep requested threads even if they exceed CPU count.")
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kind", choices=["gaussian", "random_walk", "seasonal"], default="gaussian")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-warmup", type=int, default=0, help="Warmup runs for fABBA/ABBA baselines only.")
    parser.add_argument("--baseline-repeats", type=int, default=1, help="Timed repeats for fABBA/ABBA baselines only.")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results_univariate_scaling"))
    parser.add_argument("--fontsize", type=int, default=10, help="Global figure font size for labels, ticks, legends and titles.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "svg"])
    parser.add_argument("--sample-points", type=int, default=300)
    parser.add_argument("--skip-abba", action="store_true", help="Skip ABBA if it is too slow for large n.")
    parser.add_argument("--skip-baselines", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-plots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mpi-processes",
        type=int,
        default=1,
        help=(
            "MPI processes used for JABBA digitization. The default 0 means use M=threads; "
            "outside mpirun this launches one MPI run per thread value. Use 1 to disable "
            "MPI digitization or a positive value to force a fixed MPI size."
        ),
    )
    parser.add_argument(
        "--jabba-compression-backend",
        choices=["cython", "legacy-cython", "python"],
        default="legacy-cython",
        help=(
            "Compression backend for JABBA. 'cython' is the revised fast kernel; "
            "'legacy-cython' is the original fABBA/JABBA Cython compression; "
            "'python' forces the Python/NumPy fallback."
        ),
    )
    parser.add_argument(
        "--jabba-inverse-backend",
        choices=["native", "legacy-python"],
        default="native",
        help=(
            "Inverse backend for JABBA timing. 'native' uses the optimized experiments/jabba inverse; "
            "'legacy-python' uses the original list-based reconstruction path to reproduce the old "
            "multithreading.py full-pipeline timing protocol."
        ),
    )
    parser.add_argument(
        "--jabba-speedup-metric",
        choices=["fit", "total", "compression"],
        default="fit",
        help="Which JABBA phase defines the headline jabba_speedup column and speedup panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_cpu = cpu_count()
    threads = resolve_threads(args, max_cpu)
    run_same_mpi_driver(args, threads, max_cpu)
    if args.mpi_processes == 0:
        if len(threads) != 1:
            raise SystemExit("When already running under mpirun, --mpi-processes 0 requires one --threads value.")
        args.mpi_processes = threads[0]
    maybe_spawn_mpi(args)
    comm, rank, mpi_size, _, _ = mpi_context()
    if rank == 0:
        args.outdir.mkdir(parents=True, exist_ok=True)

    if args.threads is None and 1 not in threads:
        threads.insert(0, 1)

    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": max_cpu,
        "threads": threads,
        "tol": args.tol,
        "alpha": args.alpha,
        "seed": args.seed,
        "kind": args.kind,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline_warmup": args.baseline_warmup,
        "baseline_repeats": args.baseline_repeats,
        "mpi_processes_requested": args.mpi_processes,
        "mpi_processes_actual": mpi_size,
        "jabba_compression_backend": args.jabba_compression_backend,
        "jabba_inverse_backend": args.jabba_inverse_backend,
        "jabba_speedup_metric": args.jabba_speedup_metric,
    }
    if rank == 0:
        (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows = []
    reconstruction_examples = {}
    last_series = None

    for length in args.lengths:
        if rank == 0:
            series = make_series(length, args.seed, args.kind)
            last_series = series
            print(f"\n=== length={length:,} tol={args.tol} alpha={args.alpha} mpi_processes={mpi_size} ===")

            if not args.skip_baselines:
                print("Running fABBA baseline...")
                fabba_runs, fabba_summary = timed_runs(
                    lambda: run_fabba(series, args.tol, args.alpha),
                    args.baseline_warmup,
                    args.baseline_repeats,
                )
                fabba_k = int(fabba_runs[-1]["symbols"])
                rows.append(summarize_method("fABBA", length, None, fabba_runs, fabba_summary))
                reconstruction_examples["fABBA"] = fabba_runs[-1]["reconstruction"]
                print(f"  fABBA runtime={fabba_summary['runtime_mean']:.4f}s MSE={fabba_summary['mse_mean']:.4g} symbols={fabba_k}")

                if not args.skip_abba:
                    print(f"Running ABBA baseline with k={fabba_k}...")
                    abba_runs, abba_summary = timed_runs(
                        lambda: run_abba(series, args.tol, fabba_k),
                        args.baseline_warmup,
                        args.baseline_repeats,
                    )
                    rows.append(summarize_method("ABBA", length, None, abba_runs, abba_summary))
                    reconstruction_examples["ABBA"] = abba_runs[-1]["reconstruction"]
                    print(f"  ABBA runtime={abba_summary['runtime_mean']:.4f}s MSE={abba_summary['mse_mean']:.4g}")
            else:
                print("Skipping fABBA/ABBA baselines for this MPI-size run.")
        else:
            series = None

        for t in threads:
            if rank == 0:
                print(f"Running JABBA threads=partitions={t}, mpi_processes={mpi_size}...")
            jabba_runs, jabba_summary = timed_runs(
                lambda t=t: run_jabba(
                    series,
                    args.tol,
                    args.alpha,
                    t,
                    args.jabba_compression_backend,
                    args.jabba_inverse_backend,
                ),
                args.warmup,
                args.repeats,
            )
            if rank == 0:
                rows.append(summarize_method("JABBA", length, t, jabba_runs, jabba_summary))
                reconstruction_examples["JABBA"] = jabba_runs[-1]["reconstruction"]
                print(
                    f"  JABBA-{t} total={jabba_summary['runtime_mean']:.4f}s "
                    f"fit={jabba_summary['fit_runtime_mean']:.4f}s "
                    f"comp={jabba_summary['compression_time_mean']:.4f}s "
                    f"digit={jabba_summary['digitization_time_mean']:.4f}s "
                    f"inv={jabba_summary['inverse_time_mean']:.4f}s "
                    f"MSE={jabba_summary['mse_mean']:.4g} pieces={jabba_runs[-1]['pieces']} "
                    f"symbols={jabba_runs[-1]['symbols']}"
                )

    if rank == 0:
        df = save_records(rows, args.outdir / "univariate_thread_scaling.csv", args.jabba_speedup_metric)
        if not args.skip_baselines:
            write_summary(df, args.outdir / "univariate_thread_scaling_summary.json")
        if not args.skip_plots and not args.skip_baselines:
            plot_overview(df, args.outdir, args.fontsize, args.formats)
            plot_breakdown(df, args.outdir, args.fontsize, args.formats)
            if last_series is not None and reconstruction_examples:
                plot_reconstruction(last_series, reconstruction_examples, args.outdir, args.fontsize, args.formats, args.sample_points)

        print(f"\nSaved results to {args.outdir}")
        print(f"CSV: {args.outdir / 'univariate_thread_scaling.csv'}")
        if not args.skip_plots and not args.skip_baselines:
            print(f"Overview figure: {args.outdir / 'univariate_scaling_overview.pdf'}")
            print(f"Breakdown figure: {args.outdir / 'univariate_jabba_breakdown.pdf'}")

    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":
    main()
