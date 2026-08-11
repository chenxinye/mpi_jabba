"""Aggregation adapter for revised JABBA.

The adapter prefers the standalone ``mpi-aggregation`` implementation and keeps
an API compatible with JABBA digitization. If a matching compiled MPI binding is
not importable in the current Python environment, it falls back to a local serial
implementation with the same return fields.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from .aggmem import aggregate as _cython_aggregate_raw
except Exception:  # extension is optional until setup.py build_ext --inplace runs
    _cython_aggregate_raw = None

_warned_no_mpi_aggregation = False


def _warn_no_mpi_aggregation(reason):
    global _warned_no_mpi_aggregation
    if _warned_no_mpi_aggregation:
        return
    warnings.warn(
        "mpi-aggregation backend was not used; "
        f"{reason}. Falling back to local JABBA aggregation. "
        "Build/install /Users/chenxinye/mpi_jabba/mpi-aggregation if you expect the MPI aggregation backend.",
        RuntimeWarning,
        stacklevel=3,
    )
    _warned_no_mpi_aggregation = True


_SORTING_MAP = {
    "norm": "2-norm",
    "2-norm": "2-norm",
    "1-norm": "1-norm",
    "lexi": "lexi",
}


def normalize_sorting(sorting):
    if sorting in _SORTING_MAP:
        return _SORTING_MAP[sorting]
    if sorting == "pca":
        return "pca"
    raise ValueError("sorting must be one of: norm, 2-norm, 1-norm, lexi, pca")


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _load_core_from_build():
    build = _repo_root() / "mpi-aggregation" / "build"
    for path in sorted(build.glob("_core*.so")):
        try:
            spec = importlib.util.spec_from_file_location("_core", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue
    return None


def _running_under_mpi():
    keys = (
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "PMI_SIZE",
        "PMI_RANK",
        "PMIX_RANK",
        "MPI_LOCALNRANKS",
    )
    return any(k in os.environ for k in keys)


def _load_mpi_alphaagg():
    """Load the correct standalone mpi-aggregation backend.

    The repository-local implementation is preferred over any globally installed
    package with the same import name. A globally installed ``mpi_alphaagg`` is
    used only as a final fallback, which supports workflows where
    ``/Users/chenxinye/mpi_jabba/mpi-aggregation`` has been installed with pip.
    The old ``software/qabba`` implementation is never added to ``sys.path``.
    """
    py_dir = _repo_root() / "mpi-aggregation" / "python"
    if py_dir.exists():
        py_dir_str = str(py_dir)
        if py_dir_str in sys.path:
            sys.path.remove(py_dir_str)
        sys.path.insert(0, py_dir_str)
        try:
            return importlib.import_module("mpi_alphaagg")
        except Exception:
            sys.modules.pop("mpi_alphaagg", None)

    core = _load_core_from_build()
    if core is not None:
        return core

    try:
        return importlib.import_module("mpi_alphaagg")
    except Exception:
        return None


def _sorted_indices(points, sorting):
    if sorting in {"norm", "2-norm"}:
        sort_vals = np.linalg.norm(points, ord=2, axis=1)
        return sort_vals, np.argsort(sort_vals)
    if sorting == "1-norm":
        sort_vals = np.linalg.norm(points, ord=1, axis=1)
        return sort_vals, np.argsort(sort_vals)
    if sorting == "pca":
        centered = points - points.mean(axis=0)
        if points.shape[1] > 1:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            sort_vals = centered @ vh[0]
        else:
            sort_vals = centered[:, 0]
        if sort_vals.size:
            sort_vals = sort_vals * np.sign(-sort_vals[0] if sort_vals[0] != 0 else 1.0)
        return sort_vals, np.argsort(sort_vals)
    sort_vals = np.zeros(points.shape[0], dtype=np.float64)
    return sort_vals, np.lexsort((points[:, 1], points[:, 0]))


def _format_labels_splist(points, labels, splist, backend):
    pts = np.ascontiguousarray(points, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_clusters = int(labels.max() + 1) if labels.size else 0
    clusters = np.zeros((n_clusters, 7), dtype=np.float64)
    total_sse = 0.0
    for row in splist:
        cid = int(row[1])
        mask = labels == cid
        members = pts[mask]
        centroid = members.mean(axis=0) if members.size else np.asarray(row[3:5], dtype=np.float64)
        diff = members - centroid
        sse = float(np.sum(diff * diff))
        clusters[cid] = [cid, int(row[2]), float(row[3]), float(row[4]), float(centroid[0]), float(centroid[1]), sse]
        total_sse += sse
    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "total_sse": float(total_sse),
        "clusters": clusters,
        "backend": backend,
    }


def aggregate_cython(points, alpha, sorting="2-norm"):
    """Serial Cython aggregation with mpi_alphaagg-compatible output."""
    if _cython_aggregate_raw is None:
        raise ImportError("jabba.aggmem has not been compiled")
    pts = np.ascontiguousarray(points, dtype=np.float64)
    labels, splist = _cython_aggregate_raw(pts, normalize_sorting(sorting), float(alpha))
    return _format_labels_splist(pts, labels, splist, "jabba.aggmem")


def aggregate_serial_python(points, alpha, sorting="2-norm"):
    """Serial 2D alpha aggregation with mpi_alphaagg-compatible output."""
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    sorting = normalize_sorting(sorting)
    sort_vals, order = _sorted_indices(pts, sorting)
    labels = np.full(pts.shape[0], -1, dtype=np.int64)
    cluster_rows = []
    tol2 = alpha * alpha
    label = 0

    for pos, sp in enumerate(order):
        if labels[sp] >= 0:
            continue
        seed = pts[sp]
        labels[sp] = label
        members = [sp]
        for j in order[pos:]:
            if labels[j] >= 0:
                continue
            if sorting in {"2-norm", "1-norm", "pca"} and sort_vals[j] - sort_vals[sp] > alpha:
                break
            if sorting == "lexi" and pts[j, 0] - pts[sp, 0] > alpha:
                break
            diff = seed - pts[j]
            if float(np.inner(diff, diff)) <= tol2:
                labels[j] = label
                members.append(j)
        member_pts = pts[np.asarray(members, dtype=np.int64)]
        centroid = member_pts.mean(axis=0)
        sse = float(np.sum((member_pts - centroid) ** 2))
        cluster_rows.append([label, len(members), seed[0], seed[1], centroid[0], centroid[1], sse])
        label += 1

    clusters = np.asarray(cluster_rows, dtype=np.float64)
    return {
        "labels": labels,
        "n_clusters": int(label),
        "total_sse": float(clusters[:, 6].sum()) if clusters.size else 0.0,
        "clusters": clusters.reshape((label, 7)) if label else np.zeros((0, 7), dtype=np.float64),
        "backend": "serial-python",
    }


def aggregate_points(points, alpha, sorting="2-norm", algorithm="ptga", prefer_mpi=True):
    """Aggregate points using mpi-alpha-agg or the bundled Cython fallback.

    Parameters
    ----------
    points : ndarray, shape (n, 2)
    alpha : float
    sorting : {"2-norm", "1-norm", "lexi", "norm", "pca"}
    algorithm : {"ptga", "grid", "mpi", "serial", "cython"}
    prefer_mpi : bool
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    sorting = normalize_sorting(sorting)

    if sorting == "pca" or algorithm == "cython":
        if prefer_mpi:
            reason = "PCA sorting is handled by the local backend" if sorting == "pca" else "algorithm='cython' explicitly selects the local backend"
            _warn_no_mpi_aggregation(reason)
        if _cython_aggregate_raw is not None:
            try:
                return aggregate_cython(pts, alpha, sorting)
            except Exception as exc:
                warnings.warn(f"Cython aggregation failed; falling back to Python: {exc}")
        return aggregate_serial_python(pts, alpha, sorting)

    backend = _load_mpi_alphaagg() if prefer_mpi else None
    if prefer_mpi and backend is None:
        _warn_no_mpi_aggregation("could not import the repository-local mpi_alphaagg module or its compiled _core extension")
    if backend is not None:
        use_mpi = _running_under_mpi() and algorithm not in {"serial", "cython"}
        if use_mpi and algorithm in {"ptga", "mpi", "mpi-ptga"}:
            candidates = ["aggregate_mpi_ptga", "aggregate_mpi", "aggregate_serial"]
        elif use_mpi and algorithm in {"grid", "mpi-grid"}:
            candidates = ["aggregate_mpi_grid", "aggregate_mpi", "aggregate_serial"]
        else:
            candidates = ["aggregate_serial"]

        for name in candidates:
            fn = getattr(backend, name, None)
            if fn is None:
                continue
            try:
                out = fn(pts, float(alpha), sorting)
                out = dict(out)
                out["labels"] = np.asarray(out["labels"], dtype=np.int64)
                out["clusters"] = np.asarray(out["clusters"], dtype=np.float64)
                out["n_clusters"] = int(out["n_clusters"])
                out["total_sse"] = float(out.get("total_sse", 0.0))
                out["backend"] = f"mpi-alpha-agg:{name}"
                return out
            except Exception as exc:
                warnings.warn(f"mpi-alpha-agg backend {name} failed; falling back: {exc}")
                if prefer_mpi:
                    _warn_no_mpi_aggregation(f"mpi-alpha-agg backend {name} failed with {exc!r}")
                break

    if backend is not None and prefer_mpi:
        _warn_no_mpi_aggregation("no compatible aggregate_* function was available from mpi_alphaagg")

    if _cython_aggregate_raw is not None:
        try:
            return aggregate_cython(pts, alpha, sorting)
        except Exception as exc:
            warnings.warn(f"Cython aggregation failed; falling back to Python: {exc}")
    return aggregate_serial_python(pts, alpha, sorting)

def labels_centers_from_output(out, center_kind="seed"):
    """Return labels and 2D centers from an mpi_alphaagg-style result."""
    clusters = np.asarray(out["clusters"], dtype=np.float64)
    labels = np.asarray(out["labels"], dtype=np.int64)
    if center_kind == "centroid":
        centers = clusters[:, 4:6]
    else:
        centers = clusters[:, 2:4]
    return labels, centers
