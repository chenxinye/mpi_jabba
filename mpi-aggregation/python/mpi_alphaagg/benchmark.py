from __future__ import annotations

import argparse
import csv
import os
import time
import numpy as np

from ._core import aggregate_mpi_grid, aggregate_mpi_ptga
from .datasets import make_local_dataset


def _mpi_info() -> tuple[int, int]:
    try:
        from mpi4py import MPI  # type: ignore

        comm = MPI.COMM_WORLD
        return comm.Get_rank(), comm.Get_size()
    except Exception:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
        size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
        return rank, size


def _append_csv(path: str, row: dict[str, object], write_header: bool, fieldnames: list[str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--sorting", default="2-norm")
    p.add_argument("--dataset", default="blobs", choices=["uniform", "blobs", "grid"])
    p.add_argument("--algorithm", default="ptga", choices=["ptga", "grid"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    rank, size = _mpi_info()
    points = make_local_dataset(args.n, args.dataset, args.seed, rank, size).astype(np.float64, copy=False)

    if rank == 0:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.csv)
    else:
        write_header = False

    fields = [
        "frontend",
        "algorithm",
        "n_points",
        "alpha",
        "sorting",
        "dataset",
        "seed",
        "comm_size",
        "rank_count",
        "repeat_index",
        "wall_time_sec",
        "n_local",
        "n_global_clusters",
        "total_sse",
    ]
    aggregate = aggregate_mpi_ptga if args.algorithm == "ptga" else aggregate_mpi_grid
    algorithm_name = "mpi_alphaagg_ptga" if args.algorithm == "ptga" else "mpi_alphaagg_grid"

    for r in range(args.repeat):
        t0 = time.perf_counter()
        out = aggregate(points, args.alpha, args.sorting)
        t1 = time.perf_counter()
        if rank == 0:
            row = {
                "frontend": "python",
                "algorithm": algorithm_name,
                "n_points": args.n,
                "alpha": args.alpha,
                "sorting": args.sorting,
                "dataset": args.dataset,
                "seed": args.seed,
                "comm_size": size,
                "rank_count": size,
                "repeat_index": r,
                "wall_time_sec": t1 - t0,
                "n_local": int(points.shape[0]),
                "n_global_clusters": int(out["n_clusters"]),
                "total_sse": float(out["total_sse"]),
            }
            _append_csv(args.csv, row, write_header and r == 0, fields)


if __name__ == "__main__":
    main()
