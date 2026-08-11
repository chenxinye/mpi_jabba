#!/usr/bin/env python3
"""UEA total-runtime scaling experiment for Section 8.3.

This script addresses the reviewer request:

    add an extra plot showing the total wall-clock time
    (Phase compression + digitization + inverse reconstruction)

The experiment code is intentionally a thin layer over the revised JABBA
software in ``experiments/jabba``.  Compression, digitization model state, and
reconstruction are delegated to that package:

* compression: ``JABBA.parallel_compress``;
* digitization model bookkeeping: ``Model``, ``symbolsAssign``, and
  ``JABBA.string_separation``;
* reconstruction: ``JABBA.inverse_transform``.

MPI aggregation uses the same repository-local ``mpi-aggregation`` backend and
scatter/gather helpers as the Table 3 experiment, so each MPI rank receives only
its local sorted block of compressed pieces.

Run once per MPI size and reuse the same output directory:

    cd /Users/chenxinye/mpi_jabba/revision_new
    mpirun -np 4  python uea_mpi_total_runtime_experiment.py --data-dir ../UEA2018 --outdir results_uea_total
    mpirun -np 8  python uea_mpi_total_runtime_experiment.py --data-dir ../UEA2018 --outdir results_uea_total
    mpirun -np 16 python uea_mpi_total_runtime_experiment.py --data-dir ../UEA2018 --outdir results_uea_total
    mpirun -np 32 python uea_mpi_total_runtime_experiment.py --data-dir ../UEA2018 --outdir results_uea_total

Outputs:

* ``uea_total_runtime_runs.csv``: every timed run;
* ``uea_mpi_scaling.csv``: median summary for the paper table;
* ``uea_mpi_scaling.tex``: LaTeX table body;
* ``uea_total_wallclock_bar.pdf/png``: reviewer-requested total-time plot.
* ``uea_total_wallclock_line.pdf/png``: total-time scaling line plot.
* ``uea_speedup_line.pdf/png``: speedup scaling line plot.
  Total time includes compression, digitization, inverse digitization, and
  inverse compression.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import gather_global_labels, import_mpi_alphaagg, sorted_blocks  # noqa: E402
from jabba import JABBA, Model, symbolsAssign  # noqa: E402
from jabba.aggregation import aggregate_points, labels_centers_from_output  # noqa: E402
from jabba.inverse import inv_compress, inv_digitize, quantize  # noqa: E402
from jabba.jabba import one_D_centers  # noqa: E402


DEFAULT_WORKERS_FOR_TABLE = (4, 8, 16, 32)
DATASETS = (
    #"BasicMotions",
    "CharacterTrajectories",
    "Epilepsy",
    "NATOPS",
    "StandWalkJump",
    "UWaveGestureLibrary",
)
TOLS = {
    #"BasicMotions": 0.01,
    "CharacterTrajectories": 0.01,
    "Epilepsy": 0.01,
    "NATOPS": 0.01,
    "StandWalkJump": 0.01,
    "UWaveGestureLibrary": 0.01,
}


@dataclass(frozen=True)
class RunResult:
    dataset: str
    workers: int
    repeat: int
    tol: float
    alpha: float
    size: int
    dim: int
    length: int
    num_sequences: int
    num_pieces: int
    compression_time: float
    digitization_time: float
    total_time: float
    serial_compression_time: float
    serial_digitization_time: float
    serial_inverse_digitization_time: float
    serial_inverse_compression_time: float
    serial_inverse_time: float
    serial_total_time: float
    speedup: float
    efficiency: float
    mse: float
    symbols: int
    inverse_digitization_time: float
    inverse_compression_time: float
    inverse_time: float
    compression_backend: str
    aggregation_backend: str


class SerialComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def Barrier(self) -> None:
        return None

    def bcast(self, value, root=0):
        return value

    def scatter(self, values, root=0):
        return values[0]

    def gather(self, value, root=0):
        return [value]

    def allreduce(self, value, op=None):
        return value


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
        return SerialComm(), 0, 1, time.perf_counter, None
    try:
        from mpi4py import MPI  # type: ignore
    except Exception:
        return SerialComm(), 0, 1, time.perf_counter, None
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size(), MPI.Wtime, MPI.MAX


def preprocess_arff(data) -> np.ndarray:
    time_series = []
    for sample in data[0]:
        channels = []
        for channel in sample[0]:
            channels.append(list(channel))
        time_series.append(channels)
    return np.nan_to_num(np.array(time_series).astype(np.float32))


def dataset_paths(data_dir: Path, name: str) -> tuple[Path, Path]:
    root = data_dir / name
    return root / f"{name}_TRAIN.arff", root / f"{name}_TEST.arff"


def load_uea_dataset(data_dir: Path, name: str) -> np.ndarray:
    try:
        from scipy.io import arff
    except Exception as exc:
        raise ImportError("scipy is required to read the UEA .arff files used by this experiment") from exc

    train_path, test_path = dataset_paths(data_dir, name)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing {train_path} or {test_path}")
    train = arff.loadarff(train_path)
    test = arff.loadarff(test_path)
    return np.vstack((preprocess_arff(train), preprocess_arff(test))).astype(np.float32)


def reshape_joint_dataset(multivariate_ts: np.ndarray) -> np.ndarray:
    num_samples, dim, length = multivariate_ts.shape
    return multivariate_ts.reshape(num_samples * dim, length)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mu) / std


def compute_global_mse(x: np.ndarray, x_hat: Iterable[np.ndarray]) -> float:
    values = []
    for original, reconstructed in zip(x, x_hat):
        a = np.asarray(original, dtype=np.float64)
        b = np.asarray(reconstructed, dtype=np.float64)
        n = min(len(a), len(b))
        if n:
            diff = a[:n] - b[:n]
            values.append(float(np.mean(diff * diff)))
    return float(np.mean(values)) if values else math.nan


def inverse_digitize_one(seq, centers: np.ndarray, alphabets: list, target_length: int | None) -> np.ndarray:
    pieces = np.asarray(inv_digitize(seq, centers, alphabets), dtype=np.float64)
    pieces = np.asarray(quantize(pieces), dtype=np.float64)
    if target_length is not None and len(pieces):
        target_sum = max(int(target_length) - 1, 1)
        current = int(round(np.sum(pieces[:, 0])))
        pieces[-1, 0] = max(1, pieces[-1, 0] + target_sum - current)
    return pieces


def inverse_compress_one(pieces: np.ndarray, start: float, target_length: int | None) -> np.ndarray:
    reconstructed = np.asarray(inv_compress(pieces, float(start)), dtype=np.float64)
    if target_length is not None:
        target = int(target_length)
        if len(reconstructed) > target:
            reconstructed = reconstructed[:target]
        elif len(reconstructed) < target and len(reconstructed):
            reconstructed = np.pad(reconstructed, (0, target - len(reconstructed)), mode="edge")
    return reconstructed


def threaded_map(fn, items: list[tuple], n_jobs: int) -> list:
    if not items:
        return []
    n_jobs = max(1, min(int(n_jobs), len(items)))
    if n_jobs == 1:
        return [fn(*item) for item in items]
    pool = Pool(n_jobs)
    jobs = [pool.apply_async(fn, item) for item in items]
    pool.close()
    pool.join()
    return [job.get() for job in jobs]


def split_list_for_mpi(values: list, workers: int) -> list[list]:
    indices = np.array_split(np.arange(len(values)), workers)
    return [[values[int(idx)] for idx in block] for block in indices]


def distributed_inverse_and_mse(
    model: JABBA,
    x_norm: np.ndarray | None,
    symbols,
    inverse_jobs: int,
) -> tuple[float, float, float, float]:
    comm, rank, size, wall_time, reduce_max = mpi_context()
    if rank == 0:
        assert x_norm is not None
        assert symbols is not None
        assert model.parameters is not None
        centers = np.asarray(model.parameters.centers, dtype=np.float64)
        alphabets = model.parameters.alphabets.tolist()
        start_set = list(model.start_set or [])
        target_lengths = list(model.target_lengths_ or [len(row) for row in x_norm])
        symbol_blocks = split_list_for_mpi(list(symbols), size)
        start_blocks = split_list_for_mpi(start_set, size)
        length_blocks = split_list_for_mpi(target_lengths, size)
        x_blocks = [np.ascontiguousarray(block, dtype=np.float64) for block in np.array_split(x_norm, size, axis=0)]
    else:
        centers = None
        alphabets = None
        symbol_blocks = None
        start_blocks = None
        length_blocks = None
        x_blocks = None

    centers = comm.bcast(centers, root=0)
    alphabets = comm.bcast(alphabets, root=0)
    local_symbols = comm.scatter(symbol_blocks, root=0)
    local_starts = comm.scatter(start_blocks, root=0)
    local_lengths = comm.scatter(length_blocks, root=0)
    local_x = comm.scatter(x_blocks, root=0)

    local_jobs = 1 if inverse_jobs <= 0 else max(1, int(math.ceil(inverse_jobs / size)))
    digit_items = [(seq, centers, alphabets, int(length)) for seq, length in zip(local_symbols, local_lengths)]

    comm.Barrier()
    t0 = wall_time()
    local_inverse_pieces = threaded_map(inverse_digitize_one, digit_items, local_jobs)
    comm.Barrier()
    t1 = wall_time()
    compress_items = [
        (piece, float(start), int(length))
        for piece, start, length in zip(local_inverse_pieces, local_starts, local_lengths)
    ]
    local_reconstructed = threaded_map(inverse_compress_one, compress_items, local_jobs)
    comm.Barrier()
    t2 = wall_time()

    local_inverse_digitization_time = float(t1 - t0)
    local_inverse_compression_time = float(t2 - t1)
    local_inverse_time = float(t2 - t0)
    inverse_digitization_time = comm.allreduce(local_inverse_digitization_time, op=reduce_max) if reduce_max is not None else local_inverse_digitization_time
    inverse_compression_time = comm.allreduce(local_inverse_compression_time, op=reduce_max) if reduce_max is not None else local_inverse_compression_time
    inverse_time = comm.allreduce(local_inverse_time, op=reduce_max) if reduce_max is not None else local_inverse_time

    local_sse = 0.0
    local_count = 0
    for original, reconstructed in zip(local_x, local_reconstructed):
        n = min(len(original), len(reconstructed))
        if n:
            diff = np.asarray(original[:n], dtype=np.float64) - np.asarray(reconstructed[:n], dtype=np.float64)
            local_sse += float(np.dot(diff, diff))
            local_count += int(n)

    global_sse = comm.allreduce(local_sse)
    global_count = comm.allreduce(local_count)
    mse = float(global_sse / global_count) if global_count else math.nan
    return float(inverse_digitization_time), float(inverse_compression_time), float(inverse_time), mse


def count_total_pieces(pieces: list[np.ndarray]) -> int:
    return int(sum(len(piece) for piece in pieces))


def make_model(tol: float, alpha: float, sorting: str, scl: float, center_kind: str, *, prefer_mpi: bool) -> JABBA:
    return JABBA(
        tol=tol,
        init="agg",
        alpha=alpha,
        sorting=sorting,
        scl=scl,
        verbose=0,
        prefer_mpi=prefer_mpi,
        mpi_algorithm="ptga",
        center_kind=center_kind,
        auto_digitize=False,
    )


def mpi_ptga_labels_and_centers(points: np.ndarray | None, alpha: float, sorting: str, center_kind: str):
    comm, rank, size, _, _ = mpi_context()
    if size == 1 and not running_under_mpi():
        assert points is not None
        out = aggregate_points(points, alpha, sorting, algorithm="serial", prefer_mpi=False)
        labels, centers_scaled = labels_centers_from_output(out, center_kind=center_kind)
        return labels, np.ascontiguousarray(centers_scaled, dtype=np.float64), out

    _, aggregate_mpi_ptga = import_mpi_alphaagg()

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


def digitize_with_revised_jabba_mpi(
    model: JABBA,
    series: np.ndarray | None,
    pieces: list[np.ndarray] | None,
    alphabet_set: int = 0,
):
    comm, rank, _, wall_time, reduce_max = mpi_context()
    comm.Barrier()
    t0 = wall_time()

    if rank == 0:
        assert series is not None
        assert pieces is not None
        series = np.asarray(series)
        len_ts = len(series)
        model.eta = 0.000002 if series.ndim > 1 else 0.01
        num_pieces = [len(piece) for piece in pieces]
        flat = np.vstack(pieces)[:, :2].astype(np.float64, copy=False)
        model._std = np.std(flat, axis=0)
        model._std[model._std == 0] = 1.0
        len_pieces = flat[:, 0].copy()
        scaled = flat * np.array([model.scl, 1.0]) / model._std
        max_k = np.unique(scaled[:, :2], axis=0).shape[0]
        if model.auto_digitize:
            if series.ndim > 1:
                sum_of_length = sum(len(series[i]) for i in range(len_ts))
            else:
                sum_of_length = len_ts
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
        model.aggregation_backend_ = str(out.get("backend", "mpi-alpha-agg:aggregate_mpi_ptga"))
        symbols = model.string_separation(string, num_pieces)
    else:
        symbols = None

    comm.Barrier()
    elapsed = wall_time() - t0
    elapsed = comm.allreduce(elapsed, op=reduce_max) if reduce_max is not None else elapsed
    return symbols, float(elapsed), out


def run_serial_baseline(x_norm: np.ndarray, tol: float, alpha: float, sorting: str, scl: float, center_kind: str):
    model = make_model(tol, alpha, sorting, scl, center_kind, prefer_mpi=False)
    t0 = time.perf_counter()
    pieces = model.parallel_compress(x_norm, n_jobs=1)
    t1 = time.perf_counter()
    symbols = model.digitize(x_norm, pieces, n_jobs=1)
    t2 = time.perf_counter()
    centers = np.asarray(model.parameters.centers, dtype=np.float64)
    alphabets = model.parameters.alphabets.tolist()
    inverse_digit_items = [
        (seq, centers, alphabets, int(length))
        for seq, length in zip(symbols, model.target_lengths_ or [len(row) for row in x_norm])
    ]
    inverse_pieces = threaded_map(inverse_digitize_one, inverse_digit_items, 1)
    t3 = time.perf_counter()
    inverse_compress_items = [
        (piece, float(start), int(length))
        for piece, start, length in zip(inverse_pieces, model.start_set or [], model.target_lengths_ or [len(row) for row in x_norm])
    ]
    threaded_map(inverse_compress_one, inverse_compress_items, 1)
    t4 = time.perf_counter()
    return {
        "compression_time": float(t1 - t0),
        "digitization_time": float(t2 - t1),
        "inverse_digitization_time": float(t3 - t2),
        "inverse_compression_time": float(t4 - t3),
        "inverse_time": float(t4 - t2),
        "total_time": float(t4 - t0),
        "symbols": int(model.parameters.centers.shape[0]),
        "compression_backend": model.compression_backend_,
        "aggregation_backend": model.aggregation_backend_,
    }


def split_rows_for_mpi(x_norm: np.ndarray, workers: int) -> list[np.ndarray]:
    blocks = np.array_split(np.asarray(x_norm, dtype=np.float64), workers, axis=0)
    return [np.ascontiguousarray(block, dtype=np.float64) for block in blocks]


def compress_row_block(
    block: np.ndarray,
    tol: float,
    alpha: float,
    sorting: str,
    scl: float,
    center_kind: str,
    local_jobs: int,
):
    if block.size == 0 or block.shape[0] == 0:
        return [], [], [], ""
    local_model = make_model(tol, alpha, sorting, scl, center_kind, prefer_mpi=False)
    pieces = local_model.parallel_compress(block, n_jobs=max(1, int(local_jobs)))
    return pieces, local_model.start_set, local_model.target_lengths_, str(local_model.compression_backend_)


def distributed_parallel_compress(
    model: JABBA,
    x_norm: np.ndarray | None,
    tol: float,
    alpha: float,
    sorting: str,
    scl: float,
    center_kind: str,
    compress_jobs: int,
):
    comm, rank, size, wall_time, reduce_max = mpi_context()
    if rank == 0:
        assert x_norm is not None
        blocks = split_rows_for_mpi(x_norm, size)
    else:
        blocks = None

    local_jobs = 1 if compress_jobs <= 0 else max(1, int(math.ceil(compress_jobs / size)))

    comm.Barrier()
    t0 = wall_time()
    local_block = comm.scatter(blocks, root=0)
    local_pieces, local_starts, local_lengths, local_backend = compress_row_block(
        local_block,
        tol,
        alpha,
        sorting,
        scl,
        center_kind,
        local_jobs,
    )
    gathered_pieces = comm.gather(local_pieces, root=0)
    gathered_starts = comm.gather(local_starts, root=0)
    gathered_lengths = comm.gather(local_lengths, root=0)
    gathered_backends = comm.gather(local_backend, root=0)
    comm.Barrier()
    elapsed = wall_time() - t0
    elapsed = comm.allreduce(elapsed, op=reduce_max) if reduce_max is not None else elapsed

    if rank != 0:
        return None, float(elapsed), 0

    pieces = [np.asarray(piece, dtype=np.float64) for rank_pieces in gathered_pieces for piece in rank_pieces]
    model.start_set = [float(start) for rank_starts in gathered_starts for start in rank_starts]
    model.target_lengths_ = [int(length) for rank_lengths in gathered_lengths for length in rank_lengths]
    model.return_series_univariate = False
    model.compression_backend_ = next((backend for backend in gathered_backends if backend), "unknown")
    return pieces, float(elapsed), count_total_pieces(pieces)


def serial_baseline_key(name: str, tol: float, alpha: float, sorting: str, scl: float, center_kind: str) -> str:
    return (
        f"dataset={name}|tol={tol:.17g}|alpha={alpha:.17g}|"
        f"sorting={sorting}|scl={scl:.17g}|center={center_kind}|total_with_inverse=v2"
    )


def load_serial_baselines(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_serial_baselines(path: Path, baselines: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(baselines, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def get_or_measure_serial_baseline(
    name: str,
    x_norm: np.ndarray,
    tol: float,
    alpha: float,
    sorting: str,
    scl: float,
    center_kind: str,
    baseline_path: Path | None,
):
    if baseline_path is None:
        return run_serial_baseline(x_norm, tol, alpha, sorting, scl, center_kind)

    key = serial_baseline_key(name, tol, alpha, sorting, scl, center_kind)
    baselines = load_serial_baselines(baseline_path)
    if key not in baselines:
        print(f"serial total baseline: measuring {name}", flush=True)
        baselines[key] = run_serial_baseline(x_norm, tol, alpha, sorting, scl, center_kind)
        save_serial_baselines(baseline_path, baselines)
    else:
        print(f"serial total baseline: reuse {float(baselines[key]['total_time']):.6f}s for {name}", flush=True)
    return baselines[key]


def run_mpi_once(
    name: str,
    data_dir: Path,
    alpha: float,
    compress_jobs: int,
    inverse_jobs: int,
    sorting: str,
    scl: float,
    center_kind: str,
    repeat: int,
    baseline_path: Path | None,
) -> RunResult | None:
    comm, rank, size, _, _ = mpi_context()
    tol = TOLS[name]

    if rank == 0:
        try:
            multivariate_ts = load_uea_dataset(data_dir, name)
            exists = True
        except FileNotFoundError as exc:
            print(f"skip {name}: {exc}")
            multivariate_ts = None
            exists = False
    else:
        multivariate_ts = None
        exists = None
    exists = comm.bcast(exists, root=0)
    if not exists:
        return None

    if rank == 0:
        assert multivariate_ts is not None
        x = reshape_joint_dataset(multivariate_ts)
        x_norm = normalize_rows(x).astype(np.float64)

        serial = get_or_measure_serial_baseline(name, x_norm, tol, alpha, sorting, scl, center_kind, baseline_path)
        model = make_model(tol, alpha, sorting, scl, center_kind, prefer_mpi=True)
    else:
        x_norm = None
        model = make_model(tol, alpha, sorting, scl, center_kind, prefer_mpi=True)
        serial = None

    pieces, compression_time, num_pieces = distributed_parallel_compress(
        model,
        x_norm,
        tol,
        alpha,
        sorting,
        scl,
        center_kind,
        compress_jobs,
    )

    symbols, digitization_time, _ = digitize_with_revised_jabba_mpi(model, x_norm, pieces)

    inverse_digitization_time, inverse_compression_time, inverse_time, mse = distributed_inverse_and_mse(
        model,
        x_norm,
        symbols,
        inverse_jobs,
    )

    if rank != 0:
        return None

    assert x_norm is not None
    assert pieces is not None
    assert serial is not None
    assert symbols is not None
    assert multivariate_ts is not None
    assert compression_time is not None
    assert num_pieces is not None

    total_time = float(compression_time + digitization_time + inverse_time)
    speedup = float(serial["total_time"] / total_time) if total_time > 0 else math.nan

    return RunResult(
        dataset=name,
        workers=size,
        repeat=repeat,
        tol=tol,
        alpha=alpha,
        size=int(multivariate_ts.shape[0]),
        dim=int(multivariate_ts.shape[1]),
        length=int(multivariate_ts.shape[2]),
        num_sequences=int(x_norm.shape[0]),
        num_pieces=int(num_pieces),
        compression_time=float(compression_time),
        digitization_time=float(digitization_time),
        total_time=total_time,
        serial_compression_time=float(serial["compression_time"]),
        serial_digitization_time=float(serial["digitization_time"]),
        serial_inverse_digitization_time=float(serial["inverse_digitization_time"]),
        serial_inverse_compression_time=float(serial["inverse_compression_time"]),
        serial_inverse_time=float(serial["inverse_time"]),
        serial_total_time=float(serial["total_time"]),
        speedup=speedup,
        efficiency=float(speedup / size) if size else math.nan,
        mse=float(mse),
        symbols=int(model.parameters.centers.shape[0]),
        inverse_digitization_time=float(inverse_digitization_time),
        inverse_compression_time=float(inverse_compression_time),
        inverse_time=float(inverse_time),
        compression_backend=str(model.compression_backend_),
        aggregation_backend=str(model.aggregation_backend_),
    )


def dataclass_fields() -> list[str]:
    return list(RunResult.__dataclass_fields__)


def append_csv(path: Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = dataclass_fields()
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def read_runs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "None", "nan"} else math.nan


def row_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def median_float(values: list[float]) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def fixed_serial_baselines(raw_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        grouped.setdefault(row["dataset"], []).append(row)

    baselines: dict[str, dict[str, float]] = {}
    for dataset, rows in grouped.items():
        baselines[dataset] = {
            "serial_compression_time": median_float([row_float(row, "serial_compression_time") for row in rows]),
            "serial_digitization_time": median_float([row_float(row, "serial_digitization_time") for row in rows]),
            "serial_inverse_digitization_time": median_float([row_float(row, "serial_inverse_digitization_time") for row in rows]),
            "serial_inverse_compression_time": median_float([row_float(row, "serial_inverse_compression_time") for row in rows]),
            "serial_inverse_time": median_float([row_float(row, "serial_inverse_time") for row in rows]),
            "serial_total_time": median_float([row_float(row, "serial_total_time") for row in rows]),
        }
    return baselines


def recompute_summary_speedup(row: dict[str, str], baseline: dict[str, float]) -> dict[str, str]:
    row = dict(row)
    total_time = row_float(row, "total_time")
    workers = row_int(row, "workers")
    serial_total_time = baseline["serial_total_time"]

    row["serial_compression_time"] = str(baseline["serial_compression_time"])
    row["serial_digitization_time"] = str(baseline["serial_digitization_time"])
    row["serial_inverse_digitization_time"] = str(baseline["serial_inverse_digitization_time"])
    row["serial_inverse_compression_time"] = str(baseline["serial_inverse_compression_time"])
    row["serial_inverse_time"] = str(baseline["serial_inverse_time"])
    row["serial_total_time"] = str(serial_total_time)

    speedup = serial_total_time / total_time if total_time > 0 and not math.isnan(serial_total_time) else math.nan
    row["speedup"] = str(speedup)
    row["efficiency"] = str(speedup / workers if workers > 0 and not math.isnan(speedup) else math.nan)
    return row


def median_summary_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["dataset"], row_int(row, "workers")), []).append(row)

    baselines = fixed_serial_baselines(raw_rows)
    selected = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row_float(row, "total_time"))
        chosen = dict(rows[len(rows) // 2])
        chosen = recompute_summary_speedup(chosen, baselines[chosen["dataset"]])
        selected.append(chosen)

    order = {name: idx for idx, name in enumerate(DATASETS)}
    return sorted(selected, key=lambda row: (order.get(row["dataset"], 999), row_int(row, "workers")))


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = dataclass_fields()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "--"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}g}"


def write_latex_table(path: Path, rows: list[dict[str, str]]) -> None:
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    workers_for_table = sorted({row_int(row, "workers") for row in rows}) or list(DEFAULT_WORKERS_FOR_TABLE)

    metrics = (
        ("MSE", "mse", ""),
        ("Comp. time", "compression_time", ""),
        ("Digit. time", "digitization_time", ""),
        ("Inv. digit.", "inverse_digitization_time", ""),
        ("Inv. comp.", "inverse_compression_time", ""),
        ("Inverse time", "inverse_time", ""),
        ("Total time", "total_time", ""),
        ("Speedup", "speedup", "x"),
        ("Efficiency", "efficiency", ""),
        ("Symbols", "symbols", ""),
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{MPI scaling of JABBA on selected UEA multivariate time-series datasets. Total time denotes compression, digitization, inverse digitization, and inverse compression wall-clock time.}",
        r"\label{tab:UEA_mpi_scaling_extended}",
        r"\centering",
        r"\setlength\tabcolsep{6pt}",
        r"\begin{tabular}{l l " + " ".join(["c"] * len(workers_for_table)) + "}",
        r"\toprule",
        r"Dataset & Metric & " + " & ".join(rf"$M={workers}$" for workers in workers_for_table) + r" \\",
        r"\midrule",
    ]

    nonempty = [(dataset, by_dataset[dataset]) for dataset in DATASETS if dataset in by_dataset]
    for dataset_index, (dataset, dataset_rows) in enumerate(nonempty):
        lookup = {row_int(row, "workers"): row for row in dataset_rows}
        first = dataset_rows[0]
        dataset_label = (
            rf"\multirow{{{len(metrics)}}}{{*}}{{\shortstack{{{dataset}\\"
            rf"Size={row_int(first, 'size')}, Dim={row_int(first, 'dim')}, Len={row_int(first, 'length')}\\"
            rf"($\tol={row_float(first, 'tol'):g}$)}}}}"
        )
        for metric_index, (label, key, suffix) in enumerate(metrics):
            values = []
            for workers in workers_for_table:
                row = lookup.get(workers)
                if row is None:
                    values.append("--")
                elif key == "symbols":
                    values.append(str(row_int(row, key)))
                else:
                    values.append(fmt(row_float(row, key)) + suffix)
            prefix = dataset_label if metric_index == 0 else ""
            lines.append(prefix + " & " + label + " & " + " & ".join(values) + r" \\")
        if dataset_index != len(nonempty) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_matplotlib():
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_root / "matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(tmp_root / "xdg-cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skip plots: {exc}")
        return None
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def workers_in_rows(rows: list[dict[str, str]]) -> list[int]:
    return sorted({row_int(row, "workers") for row in rows}) or list(DEFAULT_WORKERS_FOR_TABLE)


def datasets_in_rows(rows: list[dict[str, str]]) -> list[str]:
    return [dataset for dataset in DATASETS if any(row["dataset"] == dataset for row in rows)]


def plot_total_runtime(path_pdf: Path, path_png: Path, rows: list[dict[str, str]]) -> bool:
    plt = setup_matplotlib()
    if plt is None:
        return False

    datasets = datasets_in_rows(rows)
    workers_for_plot = workers_in_rows(rows)
    lookup = {(row["dataset"], row_int(row, "workers")): row for row in rows}
    x = np.arange(len(datasets), dtype=float)
    width = min(0.78 / max(len(workers_for_plot), 1), 0.16)
    offsets = (np.arange(len(workers_for_plot)) - (len(workers_for_plot) - 1) / 2.0) * width
    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(len(workers_for_plot))]
    hatches = ("", "//", "\\\\", "xx", "..", "++", "--", "oo")

    fig_width = max(7.2, 0.55 * len(datasets) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.35), constrained_layout=True)
    for idx, workers in enumerate(workers_for_plot):
        values = [
            row_float(lookup[(dataset, workers)], "total_time") if (dataset, workers) in lookup else np.nan
            for dataset in datasets
        ]
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=rf"$M={workers}$",
            color=colors[idx],
            edgecolor="black",
            linewidth=0.45,
            hatch=hatches[idx % len(hatches)],
            alpha=0.96,
        )

    ax.set_xlabel("UEA dataset")
    ax.set_ylabel("Total wall-clock time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=22, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=min(len(workers_for_plot), 6),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scaling_lines(path_pdf: Path, path_png: Path, rows: list[dict[str, str]], metric: str, ylabel: str, title: str) -> bool:
    plt = setup_matplotlib()
    if plt is None:
        return False

    datasets = datasets_in_rows(rows)
    lookup = {(row["dataset"], row_int(row, "workers")): row for row in rows}
    workers_for_plot = workers_in_rows(rows)
    cmap = plt.get_cmap("tab10")
    markers = ("o", "s", "^", "D", "v", "P", "X", "*")

    fig, ax = plt.subplots(figsize=(6.4, 3.55), constrained_layout=True)
    for idx, dataset in enumerate(datasets):
        xs = []
        ys = []
        for workers in workers_for_plot:
            row = lookup.get((dataset, workers))
            if row is None:
                continue
            value = row_float(row, metric)
            if not math.isnan(value):
                xs.append(workers)
                ys.append(value)
        if xs:
            ax.plot(
                xs,
                ys,
                marker=markers[idx % len(markers)],
                linewidth=1.8,
                markersize=5.0,
                color=cmap(idx % 10),
                label=dataset,
            )

    ax.set_xlabel("MPI processes $M$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if all(workers > 0 for workers in workers_for_plot):
        ax.set_xscale("log", base=2)
    ax.set_xticks(workers_for_plot)
    ax.set_xticklabels([str(workers) for workers in workers_for_plot])
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def write_outputs(outdir: Path) -> None:
    raw_csv = outdir / "uea_total_runtime_runs.csv"
    summary_rows = median_summary_rows(read_runs(raw_csv))
    summary_csv = outdir / "uea_mpi_scaling.csv"
    summary_tex = outdir / "uea_mpi_scaling.tex"
    write_summary_csv(summary_csv, summary_rows)
    write_latex_table(summary_tex, summary_rows)
    plot_bar_ok = plot_total_runtime(outdir / "uea_total_wallclock_bar.pdf", outdir / "uea_total_wallclock_bar.png", summary_rows)
    plot_total_line_ok = plot_scaling_lines(
        outdir / "uea_total_wallclock_line.pdf",
        outdir / "uea_total_wallclock_line.png",
        summary_rows,
        "total_time",
        "Total wall-clock time (s)",
        "Total wall-clock scaling",
    )
    plot_speedup_line_ok = plot_scaling_lines(
        outdir / "uea_speedup_line.pdf",
        outdir / "uea_speedup_line.png",
        summary_rows,
        "speedup",
        "Speedup vs serial JABBA",
        "Speedup scaling",
    )
    print(f"saved {raw_csv}")
    print(f"saved {summary_csv}")
    print(f"saved {summary_tex}")
    if plot_bar_ok:
        print(f"saved {outdir / 'uea_total_wallclock_bar.pdf'}")
        print(f"saved {outdir / 'uea_total_wallclock_bar.png'}")
    if plot_total_line_ok:
        print(f"saved {outdir / 'uea_total_wallclock_line.pdf'}")
        print(f"saved {outdir / 'uea_total_wallclock_line.png'}")
    if plot_speedup_line_ok:
        print(f"saved {outdir / 'uea_speedup_line.pdf'}")
        print(f"saved {outdir / 'uea_speedup_line.png'}")


def print_markdown(rows: list[RunResult]) -> None:
    if not rows:
        return
    headers = ("Dataset", "M", "Rep", "MSE", "Comp.", "Digit.", "Inv.", "Total", "Speedup", "Eff.", "Symbols")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            f"| {row.dataset} | {row.workers} | {row.repeat} | {fmt(row.mse)} | "
            f"{fmt(row.compression_time)} | {fmt(row.digitization_time)} | {fmt(row.inverse_time)} | {fmt(row.total_time)} | "
            f"{fmt(row.speedup)}x | {fmt(row.efficiency)} | {row.symbols} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UEA JABBA total-runtime scaling experiments.")
    parser.add_argument("--dataset", default="all", help="Dataset name or 'all'.")
    parser.add_argument("--data-dir", type=Path, default=Path("../UEA2018"))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--compress-jobs",
        type=int,
        default=0,
        help=(
            "Total local compression threads across MPI ranks. "
            "0 means distributed MPI compression with one local worker per rank."
        ),
    )
    parser.add_argument(
        "--inverse-jobs",
        type=int,
        default=-1,
        help=(
            "Total local inverse workers across MPI ranks. "
            "-1/0 means distributed MPI inverse with one local worker per rank."
        ),
    )
    parser.add_argument("--sorting", default="2-norm", choices=("2-norm", "1-norm", "lexi"))
    parser.add_argument("--scl", type=float, default=1.0)
    parser.add_argument("--center-kind", default="seed", choices=("seed", "centroid"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("results_uea_total"))
    parser.add_argument("--reset", action="store_true", help="Remove previous CSV outputs in outdir before running.")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    comm, rank, size, _, _ = mpi_context()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    unknown = [dataset for dataset in datasets if dataset not in TOLS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Known datasets: {list(DATASETS)}")

    if rank == 0:
        args.outdir.mkdir(parents=True, exist_ok=True)
        if args.reset:
            for name in (
                "uea_total_runtime_runs.csv",
                "uea_mpi_scaling.csv",
                "uea_mpi_scaling.tex",
                "uea_mpi_scaling_M4_M8_M16_M32.csv",
                "uea_mpi_scaling_M4_M8_M16_M32.tex",
                "uea_serial_total_baselines.json",
                "uea_total_wallclock_bar.pdf",
                "uea_total_wallclock_bar.png",
                "uea_total_wallclock_line.pdf",
                "uea_total_wallclock_line.png",
                "uea_speedup_line.pdf",
                "uea_speedup_line.png",
            ):
                path = args.outdir / name
                if path.exists():
                    path.unlink()
        print(f"UEA total-runtime run: M={size}, datasets={list(datasets)}, repeats={args.repeats}")

    run_rows: list[RunResult] = []
    for dataset in datasets:
        for repeat in range(1, args.repeats + 1):
            result = run_mpi_once(
                dataset,
                args.data_dir,
                args.alpha,
                args.compress_jobs,
                args.inverse_jobs,
                args.sorting,
                args.scl,
                args.center_kind,
                repeat,
                args.outdir / "uea_serial_total_baselines.json",
            )
            if rank == 0 and result is not None:
                run_rows.append(result)
                print(
                    f"{dataset} M={size} repeat={repeat}/{args.repeats} "
                    f"comp={result.compression_time:.6f}s digit={result.digitization_time:.6f}s "
                    f"inv_digit={result.inverse_digitization_time:.6f}s "
                    f"inv_comp={result.inverse_compression_time:.6f}s "
                    f"total={result.total_time:.6f}s speedup={result.speedup:.3f}x "
                    f"mse={result.mse:.6g} symbols={result.symbols} "
                    f"backend={result.compression_backend}/{result.aggregation_backend}"
                )

    if rank == 0:
        append_csv(args.outdir / "uea_total_runtime_runs.csv", run_rows)
        write_outputs(args.outdir)
        print_markdown(run_rows)
        (args.outdir / f"all_M{size}.json").write_text(
            json.dumps([row.__dict__ for row in run_rows], indent=2),
            encoding="utf-8",
        )

    comm.Barrier()


if __name__ == "__main__":
    main()
