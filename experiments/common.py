from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MPI_AGG_ROOT = REPO_ROOT / "mpi-aggregation"


def import_mpi_alphaagg():
    python_dir = MPI_AGG_ROOT / "python"
    build_dir = MPI_AGG_ROOT / "build"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))

    try:
        from mpi_alphaagg import aggregate_mpi_ptga, aggregate_serial

        return aggregate_serial, aggregate_mpi_ptga
    except Exception:
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        from _core import aggregate_mpi_ptga, aggregate_serial

        return aggregate_serial, aggregate_mpi_ptga


def sorting_values(points: np.ndarray, sorting: str) -> np.ndarray:
    if sorting == "2-norm":
        return np.linalg.norm(points, axis=1)
    if sorting == "1-norm":
        return np.abs(points).sum(axis=1)
    raise ValueError("sorting must be one of: 2-norm, 1-norm, lexi")


def sorted_blocks(points: np.ndarray, size: int, sorting: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if sorting == "lexi":
        order = np.lexsort((points[:, 1], points[:, 0]))
    else:
        order = np.argsort(sorting_values(points, sorting), kind="mergesort")
    blocks = np.array_split(order.astype(np.int64, copy=False), size)
    data_blocks = [np.ascontiguousarray(points[block], dtype=np.float64) for block in blocks]
    index_blocks = [np.ascontiguousarray(block, dtype=np.int64) for block in blocks]
    return data_blocks, index_blocks


def compute_sse(points: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    total = 0.0
    for lab in np.unique(labels):
        group = points[labels == lab]
        center = group.mean(axis=0)
        diff = group - center
        total += float(np.sum(diff * diff))
    return total


def gather_global_labels(comm, local_indices: np.ndarray, local_labels: np.ndarray, n: int) -> np.ndarray | None:
    gathered = comm.gather((local_indices, local_labels), root=0)
    if comm.Get_rank() != 0:
        return None
    labels = np.empty(n, dtype=np.int64)
    for indices, labs in gathered:
        labels[indices] = labs
    return labels


def append_csv(path: Path, row: dict[str, object], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def latest_build_hint() -> str:
    matches = glob.glob(str(MPI_AGG_ROOT / "build" / "_core*.so"))
    if matches:
        return matches[0]
    return "not built"
