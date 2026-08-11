from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from . import aggregate_cython, aggregate_serial
from .datasets import make_local_dataset


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


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
    p.add_argument("--sorting", default="2-norm", choices=["2-norm", "1-norm", "lexi"])
    p.add_argument("--dataset", default="blobs", choices=["uniform", "blobs", "grid"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--csv", required=True)
    p.add_argument("--frontend", default="python")
    args = p.parse_args()

    points = make_local_dataset(args.n, args.dataset, args.seed, rank=0, size=1).astype(np.float64, copy=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    write_header = not os.path.exists(args.csv)

    fields = [
        "frontend",
        "n_points",
        "alpha",
        "sorting",
        "dataset",
        "seed",
        "repeat_index",
        "native_wall_time_sec",
        "cython_wall_time_sec",
        "native_to_cython_time_ratio",
        "native_total_sse",
        "cython_total_sse",
        "native_to_cython_sse_ratio",
        "native_n_clusters",
        "cython_n_clusters",
    ]

    for r in range(args.repeat):
        t0 = time.perf_counter()
        native = aggregate_serial(points, args.alpha, args.sorting)
        t1 = time.perf_counter()
        cy_t0 = time.perf_counter()
        cython = aggregate_cython(points, args.alpha, args.sorting)
        cy_t1 = time.perf_counter()

        native_time = t1 - t0
        cython_time = cy_t1 - cy_t0
        native_sse = float(native["total_sse"])
        cython_sse = float(cython["total_sse"])
        row = {
            "frontend": args.frontend,
            "n_points": args.n,
            "alpha": args.alpha,
            "sorting": args.sorting,
            "dataset": args.dataset,
            "seed": args.seed,
            "repeat_index": r,
            "native_wall_time_sec": native_time,
            "cython_wall_time_sec": cython_time,
            "native_to_cython_time_ratio": _ratio(native_time, cython_time),
            "native_total_sse": native_sse,
            "cython_total_sse": cython_sse,
            "native_to_cython_sse_ratio": _ratio(native_sse, cython_sse),
            "native_n_clusters": int(native["n_clusters"]),
            "cython_n_clusters": int(cython["n_clusters"]),
        }
        should_write_header = write_header and r == 0
        _append_csv(args.csv, row, should_write_header, fields)


if __name__ == "__main__":
    main()
