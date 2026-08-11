#!python
# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport fabs

np.import_array()


cdef tuple _sort_values_and_indices(np.ndarray[np.float64_t, ndim=2] data, str sorting):
    cdef np.ndarray[np.float64_t, ndim=2] centered
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals
    if sorting == "norm" or sorting == "2-norm":
        sort_vals = np.linalg.norm(data, ord=2, axis=1).astype(np.float64)
        return sort_vals, np.argsort(sort_vals).astype(np.int64), data
    if sorting == "1-norm":
        sort_vals = np.linalg.norm(data, ord=1, axis=1).astype(np.float64)
        return sort_vals, np.argsort(sort_vals).astype(np.int64), data
    if sorting == "lexi":
        sort_vals = np.zeros(data.shape[0], dtype=np.float64)
        return sort_vals, np.lexsort((data[:, 1], data[:, 0])).astype(np.int64), data

    centered = (data - np.mean(data, axis=0)).astype(np.float64)
    if data.shape[1] > 1:
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        sort_vals = (centered @ vh[0]).astype(np.float64)
    else:
        sort_vals = centered[:, 0].astype(np.float64)
    if sort_vals.shape[0] > 0 and sort_vals[0] != 0.0:
        sort_vals = sort_vals * np.sign(-sort_vals[0])
    return sort_vals, np.argsort(sort_vals).astype(np.int64), centered


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef aggregate(object data, str sorting="2-norm", double tol=0.5):
    """Cython alpha-neighborhood aggregation compatible with JABBA digitization."""
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be two-dimensional")
    if arr.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    cdef Py_ssize_t len_ind = arr.shape[0]
    cdef Py_ssize_t fdim = arr.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals
    cdef np.ndarray[np.int64_t, ndim=1] ind
    cdef np.ndarray[np.float64_t, ndim=2] cdata_arr
    sort_vals, ind, cdata_arr = _sort_values_and_indices(arr, sorting)

    cdef double[:, ::1] cdata = np.ascontiguousarray(cdata_arr, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] labels_arr = np.full(len_ind, -1, dtype=np.int64)
    cdef long long[:] labels = labels_arr
    cdef list splist = []
    cdef Py_ssize_t i, ii, j, coord, sp
    cdef long long lab = 0
    cdef long long num_group
    cdef double dist, gap, x_gap
    cdef double tol2 = tol * tol

    for i in range(len_ind):
        sp = ind[i]
        if labels[sp] >= 0:
            continue
        labels[sp] = lab
        num_group = 1

        for ii in range(i, len_ind):
            j = ind[ii]
            if labels[j] != -1:
                continue

            if sorting == "lexi":
                x_gap = cdata[j, 0] - cdata[sp, 0]
                if x_gap > tol:
                    break
                if fabs(x_gap - tol) <= 1e-12 and cdata[j, 1] > cdata[sp, 1]:
                    break
            else:
                gap = sort_vals[j] - sort_vals[sp]
                if gap > tol:
                    break

            dist = 0.0
            for coord in range(fdim):
                dist += (cdata[sp, coord] - cdata[j, coord]) ** 2
            if dist <= tol2:
                labels[j] = lab
                num_group += 1

        splist.append([sp, lab, num_group, arr[sp, 0], arr[sp, 1]])
        lab += 1

    return np.asarray(labels_arr), splist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef aggregate_1d(double[:] data, double tol=0.5):
    cdef list splist = []
    cdef np.ndarray[np.float64_t, ndim=1] sort_vals_arr = np.asarray(data, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=1] ind = np.argsort(sort_vals_arr).astype(np.int64)
    cdef list labels = [-1] * len(sort_vals_arr)
    cdef Py_ssize_t len_ind = len(sort_vals_arr)
    cdef Py_ssize_t lab = 0
    cdef Py_ssize_t sp, i, jj, j, num_group
    cdef double clustc, dat, dist

    for i in range(len_ind):
        sp = ind[i]
        if labels[sp] >= 0:
            continue
        if data[sp] < sort_vals_arr[len_ind - 1] - tol:
            clustc = data[sp] + tol
        else:
            clustc = data[sp]
        labels[sp] = lab
        num_group = 1

        for jj in range(i, len_ind):
            j = ind[jj]
            if labels[j] >= 0:
                continue
            if fabs(sort_vals_arr[j] - clustc) > tol:
                break
            dat = clustc - data[j]
            dist = dat * dat
            if dist <= tol * tol:
                labels[j] = lab
                num_group += 1
        splist.append([sp, lab, num_group, clustc])
        lab += 1
    return np.asarray(labels, dtype=np.int64), splist
