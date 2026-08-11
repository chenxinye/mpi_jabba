"""Simple transform/inverse example for revised JABBA with timing plot."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np

from jabba import JABBA


N_JOBS_TO_TIME = (1, 2, 4, 6, 8)
WARMUP_REPEATS = 1
TIMED_REPEATS = 3
OUTPUT_DIR = Path(__file__).resolve().parent
TIMING_CSV = OUTPUT_DIR / "simple_transform_timing.csv"
TIMING_PNG = OUTPUT_DIR / "simple_transform_timing.png"


def _mpi_context():
    try:
        from mpi4py import MPI  # type: ignore

        return MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size(), MPI.Wtime
    except Exception:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
        return None, rank, size, time.perf_counter


def _max_time(comm, value):
    if comm is None:
        return float(value)
    try:
        from mpi4py import MPI  # type: ignore

        return float(comm.allreduce(value, op=MPI.MAX))
    except Exception:
        return float(value)


def _make_series():
    x = np.linspace(0.0, 8.0 * np.pi, 400)
    return np.sin(x) + 0.15 * np.cos(3.0 * x)


def _summarize_rows(rows):
    summary = []
    for n_jobs in N_JOBS_TO_TIME:
        group = [row for row in rows if row["n_jobs"] == n_jobs]
        fit = np.asarray([row["fit_transform_sec"] for row in group], dtype=np.float64)
        inv = np.asarray([row["inverse_transform_sec"] for row in group], dtype=np.float64)
        summary.append(
            {
                "n_jobs": n_jobs,
                "fit_mean": float(np.mean(fit)),
                "fit_std": float(np.std(fit, ddof=1)) if len(fit) > 1 else 0.0,
                "inverse_mean": float(np.mean(inv)),
                "inverse_std": float(np.std(inv, ddof=1)) if len(inv) > 1 else 0.0,
            }
        )
    return summary


def _plot_timing(rows, output_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is not available; skip timing plot: {exc}")
        return False

    summary = _summarize_rows(rows)
    labels = [str(row["n_jobs"]) for row in summary]
    fit_means = [row["fit_mean"] for row in summary]
    fit_stds = [row["fit_std"] for row in summary]
    inverse_means = [row["inverse_mean"] for row in summary]
    inverse_stds = [row["inverse_std"] for row in summary]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(x - width / 2, fit_means, width, yerr=fit_stds, capsize=4, label="fit_transform")
    ax.bar(x + width / 2, inverse_means, width, yerr=inverse_stds, capsize=4, label="inverse_transform")
    ax.set_xlabel("compression n_jobs")
    ax.set_ylabel("wall time (s), mean of 3 runs")
    ax.set_title("Revised JABBA timing after one warmup")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)
    return True


def run_example():
    comm, rank, size, wall_time = _mpi_context()
    ts = _make_series()
    rows = []
    last_result = None

    for n_jobs in N_JOBS_TO_TIME:
        for repeat in range(WARMUP_REPEATS + TIMED_REPEATS):
            is_warmup = repeat < WARMUP_REPEATS
            timed_repeat = repeat - WARMUP_REPEATS + 1
            show_verbose = rank == 0 and repeat == 0
            model = JABBA(tol=0.05, alpha=0.25, sorting="2-norm", verbose=1 if show_verbose else 0, prefer_mpi=True)

            t0 = wall_time()
            symbols, starts = model.fit_transform(ts, n_jobs=n_jobs, return_start_set=True)
            fit_time = _max_time(comm, wall_time() - t0)

            t0 = wall_time()
            reconstructed = model.inverse_transform(symbols, starts, n_jobs=1)
            inverse_time = _max_time(comm, wall_time() - t0)

            last_result = (model, symbols, reconstructed)

            if is_warmup:
                if rank == 0:
                    print(
                        "warmup: "
                        f"mpi_ranks={size} n_jobs={n_jobs} "
                        f"fit_transform={fit_time:.6f}s inverse={inverse_time:.6f}s "
                        f"compression={model.compression_backend_} aggregation={model.aggregation_backend_}"
                    )
                continue

            row = {
                "mpi_ranks": size,
                "n_jobs": n_jobs,
                "repeat": timed_repeat,
                "fit_transform_sec": fit_time,
                "inverse_transform_sec": inverse_time,
                "compression_backend": model.compression_backend_,
                "aggregation_backend": model.aggregation_backend_,
                "symbol_sequences": len(symbols),
                "first_sequence_length": len(symbols[0]),
                "reconstruction_length": len(reconstructed),
            }
            rows.append(row)

            if rank == 0:
                print(
                    "timing: "
                    f"mpi_ranks={size} n_jobs={n_jobs} repeat={timed_repeat}/{TIMED_REPEATS} "
                    f"fit_transform={fit_time:.6f}s inverse={inverse_time:.6f}s "
                    f"compression={model.compression_backend_} aggregation={model.aggregation_backend_}"
                )

    if rank == 0:
        assert last_result is not None
        model, symbols, reconstructed = last_result
        print("symbol sequence count:", len(symbols))
        print("first sequence length:", len(symbols[0]))
        print("reconstruction length:", len(reconstructed))
        print("compression backend:", model.compression_backend_)
        print("aggregation backend:", model.aggregation_backend_)
        print("first 12 symbols:", "".join(symbols[0][:12]))

        with TIMING_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("timing csv:", TIMING_CSV)

        print("timing summary:")
        for item in _summarize_rows(rows):
            print(
                f"  n_jobs={item['n_jobs']}: "
                f"fit_mean={item['fit_mean']:.6f}s fit_std={item['fit_std']:.6f}s "
                f"inverse_mean={item['inverse_mean']:.6f}s inverse_std={item['inverse_std']:.6f}s"
            )

        if _plot_timing(rows, TIMING_PNG):
            print("timing plot:", TIMING_PNG)

    return rows, last_result


if __name__ == "__main__":
    run_example()
