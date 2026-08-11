from __future__ import annotations

import math
import numpy as np


def local_partition(n: int, rank: int, size: int) -> tuple[int, int]:
    base = n // size
    rem = n % size
    local_n = base + (1 if rank < rem else 0)
    start = rank * base + min(rank, rem)
    return start, local_n


def make_local_dataset(n: int, dataset: str, seed: int, rank: int, size: int) -> np.ndarray:
    start, local_n = local_partition(n, rank, size)
    # Use a large prime stride so sequential rank seeds do not share nearby RNG states.
    rng = np.random.default_rng(seed + rank * 9973)
    if dataset == "uniform":
        return rng.random((local_n, 2), dtype=np.float64)
    if dataset == "blobs":
        centers = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]], dtype=np.float64)
        cid = rng.integers(0, len(centers), size=local_n)
        noise = rng.normal(0.0, 0.05, size=(local_n, 2))
        return centers[cid] + noise
    if dataset == "grid":
        side = max(1, int(math.sqrt(max(n, 1))))
        gids = np.arange(start, start + local_n, dtype=np.int64)
        gx = gids % side
        gy = gids // side
        denom = max(side - 1, 1)
        pts = np.column_stack([gx / denom, gy / denom]).astype(np.float64)
        pts += rng.normal(0.0, 0.01, size=pts.shape)
        return pts
    raise ValueError(f"unknown dataset: {dataset}")
