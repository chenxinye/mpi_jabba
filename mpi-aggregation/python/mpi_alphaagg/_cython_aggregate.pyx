#!python
# cython: language_level=3

# License: BSD 3 clause
#
# Copyright (c) 2021, Stefan Güttel, Xinye Chen
# All rights reserved.

import numpy as np
cimport numpy as np
from libc.math cimport fabs

np.import_array()

DEF EPSILON = 1e-12


cpdef aggregate(object data, str sorting, double tol=0.5):
    """Aggregate 2D data using the original fABBA Cython implementation."""
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.ascontiguousarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")

    cdef double[:, ::1] view = arr
    cdef Py_ssize_t len_ind = arr.shape[0]
    cdef double[:] sort_vals
    cdef Py_ssize_t[:] ind
    cdef Py_ssize_t sp
    cdef unsigned int lab = 0
    cdef unsigned int num_group
    cdef double[:] clustc
    cdef double dist
    cdef np.ndarray[np.int64_t, ndim=1] labels_arr = np.full(len_ind, -1, dtype=np.int64)
    cdef long long[:] labels = labels_arr
    cdef list splist = []
    cdef Py_ssize_t i, ii, j
    cdef double tol2 = tol * tol
    cdef double x_gap

    if sorting == "2-norm":
        sort_vals = np.linalg.norm(arr, ord=2, axis=1)
        ind = np.argsort(sort_vals)
    elif sorting == "1-norm":
        sort_vals = np.linalg.norm(arr, ord=1, axis=1)
        ind = np.argsort(sort_vals)
    else:
        ind = np.lexsort((arr[:, 1], arr[:, 0]), axis=0)

    for i in range(len_ind):
        sp = ind[i]

        if labels[sp] >= 0:
            continue

        clustc = view[sp, :]
        labels[sp] = lab
        num_group = 1

        for ii in range(i, len_ind):
            j = ind[ii]

            if labels[j] != -1:
                continue

            dist = (clustc[0] - view[j, 0]) ** 2
            dist += (clustc[1] - view[j, 1]) ** 2

            if dist <= tol2:
                num_group += 1
                labels[j] = lab
            else:
                if sorting == "2-norm" or sorting == "1-norm":
                    if sort_vals[j] - sort_vals[sp] > tol:
                        break
                else:
                    x_gap = view[j, 0] - view[sp, 0]
                    if x_gap > tol:
                        break
                    if (fabs(x_gap - tol) <= EPSILON) and (view[j, 1] > view[sp, 1]):
                        break

        splist.append([sp, lab, num_group, view[sp, 0], view[sp, 1]])
        lab += 1

    return np.asarray(labels_arr), splist
