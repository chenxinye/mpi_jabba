from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

from common import append_csv, compute_sse, gather_global_labels, import_mpi_alphaagg, sorted_blocks


def make_gaussian_blobs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    cid = rng.integers(0, centers.shape[0], size=n)
    return centers[cid] + 0.30 * rng.normal(size=(n, 2))


def _baseline_key(
    n: int,
    alpha: float,
    sorting: str,
    seed: int,
    warmup: int,
    repeats: int,
) -> str:
    """Return a stable key for one serial reference measurement."""
    return (
        f"alpha={alpha:.17g}|n={n}|sorting={sorting}|seed={seed}|"
        f"warmup={warmup}|repeats={repeats}"
    )


def _load_baselines(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_baselines(path: Path, baselines: dict[str, dict[str, object]]) -> None:
    """Atomically save serial baselines so later MPI sizes reuse the same reference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(baselines, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _get_or_measure_serial_baseline(
    *,
    points: np.ndarray,
    n: int,
    alpha: float,
    sorting: str,
    seed: int,
    warmup: int,
    repeats: int,
    aggregate_serial,
    baseline_path: Path,
) -> dict[str, object]:
    """
    Return one canonical serial baseline for (alpha, n, sorting, seed, timing settings).

    The first MPI-size run measures and stores the baseline. Later MPI-size runs read
    exactly the same baseline, so speedup(M) = T_serial_ref / T_mpi(M) is comparable
    across all M values.
    """
    key = _baseline_key(n, alpha, sorting, seed, warmup, repeats)
    baselines = _load_baselines(baseline_path)

    if key in baselines:
        record = baselines[key]
        print(
            f"  serial baseline: reuse {float(record['serial_time_mean']):.6f}s "
            f"from {baseline_path.name}",
            flush=True,
        )
        return record

    serial_times: list[float] = []
    serial_out = None

    print(
        f"  serial baseline: measuring once for alpha={alpha:g}, n={n} "
        f"({warmup} warmup + {repeats} timed runs)",
        flush=True,
    )

    for r in range(warmup + repeats):
        t0 = time.perf_counter()
        out = aggregate_serial(points, alpha, sorting)
        t1 = time.perf_counter()
        if r >= warmup:
            serial_times.append(t1 - t0)
            serial_out = out

    assert serial_out is not None
    serial_labels = np.asarray(serial_out["labels"], dtype=np.int64)
    serial_sse = compute_sse(points, serial_labels)

    record: dict[str, object] = {
        "serial_time_mean": float(np.mean(serial_times)),
        "serial_time_std": float(np.std(serial_times)),
        "serial_sse": float(serial_sse),
        "serial_clusters": int(serial_out["n_clusters"]),
    }
    baselines[key] = record
    _save_baselines(baseline_path, baselines)

    print(
        f"  serial baseline: saved {float(record['serial_time_mean']):.6f}s "
        f"to {baseline_path.name}",
        flush=True,
    )
    return record


def run_case(
    n: int,
    alpha: float,
    sorting: str,
    seed: int,
    warmup: int,
    repeats: int,
    baseline_path: Path,
) -> dict[str, object] | None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    aggregate_serial, aggregate_mpi_ptga = import_mpi_alphaagg()

    if rank == 0:
        points = make_gaussian_blobs(n, seed)
        serial_ref = _get_or_measure_serial_baseline(
            points=points,
            n=n,
            alpha=alpha,
            sorting=sorting,
            seed=seed,
            warmup=warmup,
            repeats=repeats,
            aggregate_serial=aggregate_serial,
            baseline_path=baseline_path,
        )
        data_blocks, index_blocks = sorted_blocks(points, size, sorting)
    else:
        points = None
        serial_ref = None
        data_blocks = None
        index_blocks = None

    local_points = comm.scatter(data_blocks, root=0)
    local_indices = comm.scatter(index_blocks, root=0)

    for _ in range(warmup):
        comm.Barrier()
        _ = aggregate_mpi_ptga(local_points, alpha, sorting)

    mpi_times: list[float] = []
    final_out = None
    final_labels = None

    for repeat_idx in range(repeats):
        if rank == 0:
            print(
                f"  MPI M={size}: timed repeat {repeat_idx + 1}/{repeats}",
                flush=True,
            )

        comm.Barrier()
        t0 = MPI.Wtime()
        out = aggregate_mpi_ptga(local_points, alpha, sorting)
        comm.Barrier()
        t1 = MPI.Wtime()

        mpi_times.append(comm.allreduce(t1 - t0, op=MPI.MAX))
        labels = gather_global_labels(
            comm,
            local_indices,
            np.asarray(out["labels"], dtype=np.int64),
            n,
        )
        if rank == 0:
            final_out = out
            final_labels = labels

    if rank != 0:
        return None

    assert points is not None
    assert serial_ref is not None
    assert final_out is not None
    assert final_labels is not None

    mpi_sse = compute_sse(points, final_labels)
    serial_time = float(serial_ref["serial_time_mean"])
    serial_time_std = float(serial_ref["serial_time_std"])
    serial_sse = float(serial_ref["serial_sse"])
    serial_clusters = int(serial_ref["serial_clusters"])
    mpi_time = float(np.mean(mpi_times))
    speedup = serial_time / mpi_time if mpi_time > 0 else np.nan

    return {
        "alpha": alpha,
        "n": n,
        "method": "MPI two-stage PTGA",
        "workers": size,
        "seed": seed,
        "sorting": sorting,
        # IMPORTANT: this is the SAME canonical serial reference for every M.
        "serial_time_mean": serial_time,
        "serial_time_std": serial_time_std,
        "mpi_time_mean": mpi_time,
        "mpi_time_std": float(np.std(mpi_times)),
        "speedup": speedup,
        "efficiency": speedup / size,
        "serial_sse": serial_sse,
        "mpi_sse": mpi_sse,
        "sse_ratio": mpi_sse / serial_sse if serial_sse > 0 else np.nan,
        "serial_clusters": serial_clusters,
        "mpi_clusters": int(final_out["n_clusters"]),
        "repeats": repeats,
        "warmup": warmup,
    }


def write_latex_table(csv_path: Path, tex_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.sort_values(["alpha", "n", "workers"])
    lines = [
        r"\begin{tabular}{c c c c c c c c c}",
        r"\toprule",
        r"$\alpha$ & $N$ & Method & $M$ & Time (s) & Speedup & SSE ratio & \#Clusters (Serial) & \#Clusters (MPI)\\",
        r"\midrule",
    ]

    for (alpha, n), group in df.groupby(["alpha", "n"], sort=True):
        # All rows in the group should now carry the identical canonical serial baseline.
        serial_ref = float(group["serial_time_mean"].iloc[0])
        serial_clusters = int(group["serial_clusters"].iloc[0])

        # Recompute speedup from the displayed serial reference as a consistency check.
        # This guarantees the LaTeX table is internally self-consistent even if a CSV
        # was edited or concatenated externally.
        lines.append(
            f"{alpha:g} & {int(n)} & Serial GA & -- & {serial_ref:.6f} & "
            f"1.000 & 1.0000 & {serial_clusters:,} & -- \\\\"
        )

        for _, row in group.iterrows():
            mpi_time = float(row.mpi_time_mean)
            speedup = serial_ref / mpi_time if mpi_time > 0 else np.nan
            lines.append(
                f" & & MPI two-stage & {int(row.workers)} & {mpi_time:.6f} & "
                f"{speedup:.3f} & {row.sse_ratio:.4f} & {serial_clusters:,} & "
                f"{int(row.mpi_clusters):,} \\\\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[100000, 500000])
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.1, 0.05])
    parser.add_argument("--sorting", default="2-norm", choices=["2-norm", "1-norm", "lexi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=Path("experiments/results_table2"))
    parser.add_argument(
        "--csv-name",
        default=None,
        help="CSV filename inside outdir; defaults to synth_aggregation.csv.",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        args.outdir.mkdir(parents=True, exist_ok=True)

        # The scaling shell script removes M*.csv at the beginning of a fresh
        # batch. If none exist, treat this invocation as the first M of a new batch and
        # reset the persisted serial reference to avoid reusing stale timings.
        baseline_path = args.outdir / "serial_baselines.json"
        has_previous_m_csv = any(args.outdir.glob("M*.csv"))
        if not has_previous_m_csv and baseline_path.exists():
            baseline_path.unlink()
            print(f"Reset stale serial baseline: {baseline_path}", flush=True)
    else:
        baseline_path = None

    # Every rank receives the same path string; only rank 0 reads/writes the file.
    baseline_path_str = comm.bcast(str(baseline_path) if rank == 0 else None, root=0)
    baseline_path = Path(baseline_path_str)

    rows = []
    csv_path = args.outdir / (args.csv_name or "synth_aggregation.csv")
    json_path = args.outdir / f"rank{comm.Get_size()}.json"
    tex_path = args.outdir / "synth_aggregation.tex"

    fields = [
        "alpha",
        "n",
        "method",
        "workers",
        "seed",
        "sorting",
        "serial_time_mean",
        "serial_time_std",
        "mpi_time_mean",
        "mpi_time_std",
        "speedup",
        "efficiency",
        "serial_sse",
        "mpi_sse",
        "sse_ratio",
        "serial_clusters",
        "mpi_clusters",
        "repeats",
        "warmup",
    ]

    total_cases = len(args.alpha) * len(args.n)
    case_idx = 0

    for alpha in args.alpha:
        for n in args.n:
            case_idx += 1
            if rank == 0:
                print(
                    f"\n[{case_idx}/{total_cases}] alpha={alpha:g}, n={n}, M={comm.Get_size()}",
                    flush=True,
                )

            row = run_case(
                n,
                alpha,
                args.sorting,
                args.seed,
                args.warmup,
                args.repeats,
                baseline_path,
            )
            if rank == 0 and row is not None:
                rows.append(row)
                append_csv(csv_path, row, fields)
                print(
                    f"  result: mpi_time={row['mpi_time_mean']:.6f}s "
                    f"serial_ref={row['serial_time_mean']:.6f}s "
                    f"speedup={row['speedup']:.3f} "
                    f"efficiency={row['efficiency']:.3f} "
                    f"sse_ratio={row['sse_ratio']:.4f}",
                    flush=True,
                )

    if rank == 0:
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        write_latex_table(csv_path, tex_path)
        print(f"\nsaved {csv_path}", flush=True)
        print(f"saved {tex_path}", flush=True)
        print(f"serial baseline file: {baseline_path}", flush=True)


if __name__ == "__main__":
    main()
