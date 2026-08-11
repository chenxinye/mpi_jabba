#!python
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def inv_transform(object strings, np.ndarray[np.float64_t, ndim=2] centers, list alphabets, double start=0, object target_length=None):
    cdef np.ndarray[np.float64_t, ndim=2] pieces = np.asarray(inv_digitize(strings, centers, alphabets), dtype=np.float64)
    pieces = np.asarray(quantize(pieces), dtype=np.float64)
    cdef long target_sum, current
    if target_length is not None and pieces.shape[0] > 0:
        target_sum = max(int(target_length) - 1, 1)
        current = int(round(float(np.sum(pieces[:, 0]))))
        pieces[pieces.shape[0] - 1, 0] = max(1.0, pieces[pieces.shape[0] - 1, 0] + target_sum - current)
    return inv_compress(pieces, start)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inv_digitize(object strings, np.ndarray[np.float64_t, ndim=2] centers, list alphabets):
    cdef list symbols = list(strings)
    cdef Py_ssize_t n = len(symbols)
    cdef np.ndarray[np.float64_t, ndim=2] pieces = np.empty((n, 2), dtype=np.float64)
    cdef Py_ssize_t i
    cdef int idx
    for i in range(n):
        idx = alphabets.index(symbols[i])
        pieces[i, 0] = centers[idx, 0]
        pieces[i, 1] = centers[idx, 1]
    return pieces


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef quantize(np.ndarray[np.float64_t, ndim=2] pieces):
    cdef np.ndarray[np.float64_t, ndim=2] out = np.array(pieces, dtype=np.float64, copy=True)
    cdef Py_ssize_t p
    cdef double corr
    if out.shape[0] == 0:
        return out
    if out.shape[0] == 1:
        out[0, 0] = round(out[0, 0])
        return out
    for p in range(out.shape[0] - 1):
        corr = round(out[p, 0]) - out[p, 0]
        out[p, 0] = round(out[p, 0] + corr)
        out[p + 1, 0] = out[p + 1, 0] - corr
        if out[p, 0] == 0:
            out[p, 0] = 1
            out[p + 1, 0] -= 1
    out[out.shape[0] - 1, 0] = round(out[out.shape[0] - 1, 0], 0)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inv_compress(np.ndarray[np.float64_t, ndim=2] pieces, double start):
    cdef list time_series = [start]
    cdef Py_ssize_t j, i
    cdef long length
    cdef double inc, base, value
    for j in range(pieces.shape[0]):
        length = max(int(round(pieces[j, 0])), 1)
        inc = pieces[j, 1]
        base = time_series[len(time_series) - 1]
        for i in range(1, length + 1):
            value = base + (float(i) / float(length)) * inc
            time_series.append(value)
    return time_series
