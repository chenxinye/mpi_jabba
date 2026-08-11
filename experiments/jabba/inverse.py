"""Inverse digitization and decompression helpers."""

from __future__ import annotations

import numpy as np

try:
    from . import inversetc as _cython_inverse
except Exception:  # extension is optional until setup.py build_ext --inplace runs
    _cython_inverse = None


def inv_digitize(strings, centers, alphabets):
    """Convert symbolic strings back to compressed pieces."""
    if _cython_inverse is not None:
        return np.asarray(_cython_inverse.inv_digitize(strings, np.asarray(centers, dtype=np.float64), list(alphabets)), dtype=np.float64)
    if isinstance(strings, str):
        symbols = list(strings)
    else:
        symbols = list(strings)
    alphabet_list = list(alphabets)
    return np.vstack([centers[alphabet_list.index(symbol)][:2] for symbol in symbols])


def quantize(pieces):
    """Realign piece lengths with the integer grid."""
    if _cython_inverse is not None:
        return np.asarray(_cython_inverse.quantize(np.asarray(pieces, dtype=np.float64)), dtype=np.float64)
    pieces = np.asarray(pieces, dtype=np.float64).copy()
    if len(pieces) == 0:
        return pieces
    if len(pieces) == 1:
        pieces[0, 0] = round(pieces[0, 0])
        return pieces

    for p in range(len(pieces) - 1):
        corr = round(pieces[p, 0]) - pieces[p, 0]
        pieces[p, 0] = round(pieces[p, 0] + corr)
        pieces[p + 1, 0] = pieces[p + 1, 0] - corr
        if pieces[p, 0] == 0:
            pieces[p, 0] = 1
            pieces[p + 1, 0] -= 1
    pieces[-1, 0] = round(pieces[-1, 0], 0)
    return pieces


def inv_compress(pieces, start):
    """Reconstruct a numeric time series from compressed pieces."""
    if _cython_inverse is not None:
        return _cython_inverse.inv_compress(np.asarray(pieces, dtype=np.float64), float(start))
    pieces = np.asarray(pieces, dtype=np.float64)
    time_series = [float(start)]
    for length, inc in pieces[:, :2]:
        length = max(int(round(length)), 1)
        x = np.arange(0, length + 1, dtype=np.float64) / length * inc
        y = time_series[-1] + x
        time_series.extend(y[1:].tolist())
    return time_series


def inv_transform(strings, centers, alphabets, start=0, target_length=None):
    """Convert ABBA/JABBA symbolic representation back to a time series.

    If ``target_length`` is supplied, the final piece length is adjusted after
    quantization so fitted sequences can reconstruct to their original grid
    length even when clustered representative lengths are approximate.
    """
    if _cython_inverse is not None:
        return _cython_inverse.inv_transform(
            strings, np.asarray(centers, dtype=np.float64), list(alphabets), float(start), target_length
        )
    pieces = inv_digitize(strings, centers, alphabets)
    pieces = quantize(pieces)
    if target_length is not None and len(pieces):
        target_sum = max(int(target_length) - 1, 1)
        current = int(round(np.sum(pieces[:, 0])))
        pieces[-1, 0] = max(1, pieces[-1, 0] + target_sum - current)
    return inv_compress(pieces, start)
