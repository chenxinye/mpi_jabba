#!python
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t _compress_nogil(double[::1] ts, Py_ssize_t len_t, double tol, double max_len, double* out) noexcept nogil:
    cdef Py_ssize_t start = 0
    cdef Py_ssize_t end = 1
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef double epsilon = 2.220446049250313e-16
    cdef double t_st, inc, lastinc = 0.0, err, lasterr = 0.0
    cdef double pred, diff

    while end < len_t:
        t_st = ts[start]
        inc = ts[end] - t_st
        err = 0.0
        for i in range(end - start + 1):
            pred = t_st + (inc / (end - start)) * i
            diff = pred - ts[start + i]
            err += diff * diff

        if (err <= tol * (end - start - 1) + epsilon) and (end - start - 1 < max_len):
            lastinc = inc
            lasterr = err
            end += 1
        else:
            out[3 * count] = <double>(end - start - 1)
            out[3 * count + 1] = lastinc
            out[3 * count + 2] = lasterr
            count += 1
            start = end - 1

    out[3 * count] = <double>(end - start - 1)
    out[3 * count + 1] = lastinc
    out[3 * count + 2] = lasterr
    return count + 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compress(double[::1] ts, double tol=0.5, double max_len=np.finfo(float).max):
    """Cython implementation of JABBA/fABBA piecewise-linear compression.

    The expensive scan runs without the Python GIL and writes directly into a
    preallocated NumPy buffer. This avoids a serialized post-processing copy
    when several Python threads compress different partitions concurrently.
    """
    cdef Py_ssize_t len_t = ts.shape[0]
    cdef Py_ssize_t count = 0
    cdef np.ndarray[np.float64_t, ndim=2] pieces
    cdef double[:, ::1] view

    if len_t < 2:
        return np.empty((0, 3), dtype=np.float64)

    pieces = np.empty((len_t, 3), dtype=np.float64)
    view = pieces
    with nogil:
        count = _compress_nogil(ts, len_t, tol, max_len, &view[0, 0])
    return pieces[:count, :]
