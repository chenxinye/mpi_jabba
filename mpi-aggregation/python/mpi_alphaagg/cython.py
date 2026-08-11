from __future__ import annotations

import numpy as np

from ._cython_aggregate import aggregate as aggregate_cython_raw


def _validate_points(points: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return arr


def aggregate_cython(points: np.ndarray, alpha: float, sorting: str = "2-norm") -> dict[str, object]:
    """Aggregate 2D points with the bundled fABBA Cython implementation."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if sorting not in {"2-norm", "1-norm", "lexi"}:
        raise ValueError("sorting must be one of: 2-norm, 1-norm, lexi")

    pts = _validate_points(points)
    labels, splist = aggregate_cython_raw(pts, sorting, alpha)
    labels = np.asarray(labels, dtype=np.int64)
    n_clusters = int(labels.max() + 1) if labels.size else 0

    clusters = np.zeros((n_clusters, 7), dtype=np.float64)
    total_sse = 0.0
    for _, label, count, seed_x, seed_y in splist:
        cid = int(label)
        mask = labels == cid
        members = pts[mask]
        centroid = members.mean(axis=0)
        diff = members - centroid
        sse = float(np.sum(diff * diff))
        clusters[cid] = [
            float(cid),
            float(count),
            float(seed_x),
            float(seed_y),
            float(centroid[0]),
            float(centroid[1]),
            sse,
        ]
        total_sse += sse

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "total_sse": float(total_sse),
        "clusters": clusters,
    }
