"""Compression utilities for the revised JABBA implementation."""

from __future__ import annotations

import warnings

import numpy as np

try:
    from .compmem import compress as _cython_compress
except Exception:  # extension is optional until setup.py build_ext --inplace runs
    _cython_compress = None

_warned_no_cython_compress = False


def compression_backend():
    """Return the compression backend selected in this Python process."""
    if _cython_compress is not None:
        return "jabba.compmem"
    return "python-numpy"


def fillna(series, method="ffill"):
    """Fill NaN values in a 1D time series."""
    arr = np.asarray(series, dtype=np.float64).copy()
    mask = np.isnan(arr)
    if not mask.any():
        return arr

    valid = arr[~mask]
    if method == "mean" and valid.size:
        arr[mask] = float(np.mean(valid))
    elif method == "median" and valid.size:
        arr[mask] = float(np.median(valid))
    elif method == "ffill":
        for i in np.where(mask)[0]:
            arr[i] = arr[i - 1] if i > 0 else 0.0
    elif method == "bfill":
        for i in sorted(np.where(mask)[0], reverse=True):
            arr[i] = arr[i + 1] if i + 1 < len(arr) else 0.0
    else:
        arr[mask] = 0.0
    return arr


def compress(ts, tol=0.5, max_len=np.inf):
    """Approximate a time series by continuous piecewise-linear pieces.

    Each returned piece is ``[length, increment, squared_error]`` and follows
    the original JABBA/fABBA compression convention. A compiled Cython kernel is
    used when available; otherwise this function falls back to NumPy/Python.
    """
    ts = np.asarray(ts, dtype=np.float64)
    global _warned_no_cython_compress
    if _cython_compress is not None:
        return _cython_compress(np.ascontiguousarray(ts), float(tol), float(max_len))
    if not _warned_no_cython_compress:
        warnings.warn(
            "jabba.compmem Cython compression is not available; using the slower Python/NumPy compression fallback. "
            "Run `python setup.py build_ext --inplace` or `python -m pip install -e . --no-build-isolation` "
            "from /Users/chenxinye/mpi_jabba/revision_new to enable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned_no_cython_compress = True
    if ts.ndim != 1:
        raise ValueError("compress expects a one-dimensional time series")
    if len(ts) < 2:
        return []

    start = 0
    end = 1
    pieces = []
    x = np.arange(0, len(ts), dtype=np.float64)
    epsilon = np.finfo(float).eps
    lastinc = ts[1] - ts[0]
    lasterr = 0.0

    while end < len(ts):
        inc = ts[end] - ts[start]
        err_vec = ts[start] + (inc / (end - start)) * x[0 : end - start + 1] - ts[start : end + 1]
        err = float(np.inner(err_vec, err_vec))

        if (err <= tol * (end - start - 1) + epsilon) and (end - start - 1 < max_len):
            lastinc, lasterr = inc, err
            end += 1
        else:
            pieces.append([end - start - 1, lastinc, lasterr])
            start = end - 1

    pieces.append([end - start - 1, lastinc, lasterr])
    return pieces
