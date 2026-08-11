"""Revised JABBA using original multithreaded compression and MPI aggregation."""

from __future__ import annotations

import collections
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool as Pool

import numpy as np

from .aggregation import aggregate_points, labels_centers_from_output
from .compression import compress, compression_backend, fillna
from .inverse import inv_transform

try:
    from joblib import parallel_backend
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency path
    parallel_backend = None
    KMeans = None


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1


def symbolsAssign(clusters, alphabet_set=0):
    """Assign printable symbols to integer cluster labels."""
    if alphabet_set == 0:
        alphabets = list("AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz")
    elif alphabet_set == 1:
        alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    elif isinstance(alphabet_set, list) and alphabet_set:
        alphabets = alphabet_set
    else:
        alphabets = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    clusters = np.asarray(clusters, dtype=np.int64)
    n_clusters = len(np.unique(clusters))
    if n_clusters >= len(alphabets):
        alphabets = [chr(i + 33) for i in range(n_clusters)]
    else:
        alphabets = alphabets[:n_clusters]
    alphabets = np.asarray(alphabets)
    return alphabets[clusters].tolist(), alphabets


@dataclass
class Model:
    centers: np.ndarray
    alphabets: np.ndarray


class JABBA(object):
    """JABBA model with MPI-backed alpha aggregation digitization.

    The public methods follow the original JABBA interface: ``fit``,
    ``fit_transform``, ``transform``, ``inverse_transform``, ``compress`` and
    ``digitize``. Compression uses the original multithreaded partitioning
    strategy, while ``init='agg'`` digitization calls the standalone
    ``mpi-aggregation`` implementation when it is importable.
    """

    def __init__(
        self,
        tol=0.2,
        init="agg",
        k=2,
        r=0.5,
        alpha=None,
        sorting="norm",
        scl=1,
        max_iter=2,
        partition_rate=None,
        partition=None,
        max_len=np.inf,
        verbose=1,
        random_state=2022,
        fillna="ffill",
        auto_digitize=False,
        mpi_algorithm="ptga",
        prefer_mpi=True,
        center_kind="seed",
    ):
        self.tol = tol
        self.alpha = alpha
        self.k = k
        self.scl = scl
        self.init = init
        if self.alpha is None:
            auto_digitize = True
        self.max_len = max_len
        self.max_iter = max_iter
        self.sorting = sorting
        self.verbose = verbose
        self.r = r
        self.partition = partition
        self.partition_rate = partition_rate
        self.temp_symbols = None
        self.symbols_ = None
        self.auto_digitize = auto_digitize
        self.d_norm = None
        self.d_shape = None
        self.eta = None
        self.fillna = fillna
        self.random_state = random_state
        self.mpi_algorithm = mpi_algorithm
        self.prefer_mpi = prefer_mpi
        self.center_kind = center_kind
        self.parameters = None
        self.pieces = None
        self.string_ = None
        self.start_set = None
        self.target_lengths_ = None
        self.return_series_univariate = False
        self.aggregation_backend_ = None
        self.compression_backend_ = None

    def fit_transform(self, series, n_jobs=-1, alphabet_set=0, return_start_set=False):
        self.fit(series, n_jobs=n_jobs, alphabet_set=alphabet_set)
        if return_start_set:
            return self.string_, self.start_set
        return self.string_

    def fit(self, series, n_jobs=-1, alphabet_set=0):
        self.pieces = self.parallel_compress(series, n_jobs=n_jobs)
        self.string_ = self.digitize(series, self.pieces, alphabet_set=alphabet_set, n_jobs=n_jobs)
        return self

    def compress(self, series, n_jobs=-1):
        return self.parallel_compress(series, n_jobs=n_jobs)

    def parallel_compress(self, series, n_jobs=-1):
        series = np.asarray(series, dtype=np.float64)
        if series.ndim == 0:
            raise ValueError("series must be one- or two-dimensional")
        len_ts = len(series)
        n_jobs = self.n_jobs_init(n_jobs, _max=len_ts)

        if series.ndim == 1:
            self.return_series_univariate = True
            if self.partition is None:
                if self.partition_rate is None:
                    partition = n_jobs
                else:
                    partition = int(np.round(np.exp(1 / self.partition_rate), 0)) * n_jobs
                    partition = min(partition, len_ts)
            else:
                partition = int(min(self.partition, len_ts))
                n_jobs = min(n_jobs, partition)
            partition = max(int(partition), 1)
            interval = int(len_ts / partition)
            if interval <= 0:
                interval = len_ts
                partition = 1
            blocks = [series[i * interval : (i + 1) * interval] for i in range(partition - 1)]
            blocks.append(series[(partition - 1) * interval :])
            series_blocks = [block for block in blocks if len(block) >= 2]
            if self.verbose:
                if partition != 1:
                    print(f"Partition series into {len(series_blocks)} parts")
                print(f"Init {n_jobs} processors.")
        else:
            self.return_series_univariate = False
            series_blocks = [np.asarray(ts, dtype=np.float64) for ts in series if len(ts) >= 2]

        self.start_set = [float(ts[0]) for ts in series_blocks]
        self.target_lengths_ = [int(len(ts)) for ts in series_blocks]
        self.compression_backend_ = compression_backend()
        if n_jobs != 1 and len(series_blocks) > 1:
            pool = Pool(n_jobs)
            jobs = [pool.apply_async(compress, args=(fillna(ts, self.fillna), self.tol, self.max_len)) for ts in series_blocks]
            pool.close()
            pool.join()
            return [job.get() for job in jobs]
        return [compress(fillna(ts, self.fillna), self.tol, self.max_len) for ts in series_blocks]

    def digitize(self, series, pieces, alphabet_set=0, n_jobs=-1):
        series = np.asarray(series)
        len_ts = len(series)
        if series.ndim > 1:
            sum_of_length = sum(len(series[i]) for i in range(len_ts))
            self.eta = 0.000002
        else:
            sum_of_length = len_ts
            self.eta = 0.01

        num_pieces = [len(p) for p in pieces]
        flat = np.vstack(pieces)[:, :2]
        self._std = np.std(flat, axis=0)
        self._std[self._std == 0] = 1
        len_pieces = flat[:, 0].copy()
        scaled = flat * np.array([self.scl, 1.0]) / self._std
        max_k = np.unique(scaled[:, :2], axis=0).shape[0]

        if self.init in {"agg", "mpi", "mpi-agg", "mpi-ptga", "mpi-grid", "serial"}:
            if self.auto_digitize:
                denom = max(max_k * (self.eta**2) * (3 * (sum_of_length**4) + 2 - 5 * (max_k**2)), np.finfo(float).eps)
                numer = 60 * sum_of_length * max(sum_of_length - max_k, 1) * (self.tol**2)
                self.alpha = pow(numer / denom, 1 / 4)
            algorithm = self.mpi_algorithm
            if self.init == "mpi-grid":
                algorithm = "grid"
            elif self.init == "serial":
                algorithm = "serial"
            out = aggregate_points(scaled[:, :2], self.alpha, self.sorting, algorithm=algorithm, prefer_mpi=self.prefer_mpi)
            labels, centers_scaled = labels_centers_from_output(out, center_kind=self.center_kind)
            centers = centers_scaled * self._std / np.array([self.scl, 1.0])
            self.k = centers.shape[0]
            self.aggregation_backend_ = out.get("backend", "unknown")
        elif self.init in {"kmeans", "f-kmeans"}:
            if KMeans is None:
                raise ImportError("scikit-learn is required for kmeans digitization")
            if self.k > max_k:
                self.k = max_k
                warnings.warn("k is larger than the unique pieces size, so k reduces to unique pieces size.")
            km = KMeans(n_clusters=self.k, random_state=self.random_state, n_init=1, max_iter=max(10, self.max_iter))
            if parallel_backend is not None:
                with parallel_backend("threading", n_jobs=n_jobs):
                    labels = km.fit_predict(scaled[:, :2])
            else:
                labels = km.fit_predict(scaled[:, :2])
            centers = km.cluster_centers_ * self._std / np.array([self.scl, 1.0])
            self.aggregation_backend_ = "sklearn-kmeans"
        else:
            raise ValueError("init must be one of agg, mpi, mpi-agg, mpi-ptga, mpi-grid, serial, kmeans, f-kmeans")

        if self.scl == 0:
            centers[:, 0] = one_D_centers(len_pieces, labels, self.k)

        string, alphabets = symbolsAssign(labels, alphabet_set)
        self.parameters = Model(np.asarray(centers, dtype=np.float64), alphabets)
        self.num_grp = self.parameters.centers.shape[0]
        if self.verbose:
            print(f"Generate {self.num_grp} symbols")
            print(f"Compression backend: {self.compression_backend_}")
            print(f"Aggregation backend: {self.aggregation_backend_}")
        return self.string_separation(string, num_pieces)

    digitizate = digitize

    def transform(self, series, n_jobs=-1):
        series = np.asarray(series, dtype=np.float64)
        n_jobs = self.n_jobs_init(n_jobs)
        if series.ndim == 1:
            self.return_series_univariate = True
            interval = int(series.shape[0] / n_jobs) if n_jobs > 0 else series.shape[0]
            if interval > 0 and n_jobs > 1 and series.shape[0] % n_jobs == 0:
                series_blocks = [series[i * interval : (i + 1) * interval] for i in range(n_jobs)]
            else:
                series_blocks = [series]
        else:
            self.return_series_univariate = False
            series_blocks = [ts for ts in series]

        start_set = [float(ts[0]) for ts in series_blocks]
        if n_jobs != 1 and len(series_blocks) > 1:
            pool = Pool(n_jobs)
            jobs = [pool.apply_async(self.transform_single_series, args=(ts,)) for ts in series_blocks]
            pool.close()
            pool.join()
            return [job.get() for job in jobs], start_set
        return [self.transform_single_series(ts) for ts in series_blocks], start_set

    def transform_single_series(self, series):
        self.compression_backend_ = compression_backend()
        pieces = np.asarray(compress(series, self.tol, self.max_len), dtype=np.float64)[:, :2]
        return [self.piece_to_symbol(piece) for piece in pieces]

    def inverse_transform(self, string_sequences, start_set=None, n_jobs=1):
        if self.parameters is None:
            raise ValueError("Please fit the model before inverse_transform.")
        n_jobs = self.n_jobs_init(n_jobs)
        if start_set is None:
            start_set = self.start_set
        if start_set is None:
            raise ValueError("Please input valid start_set.")

        if isinstance(string_sequences, (str, tuple)):
            string_sequences = [list(string_sequences)]
        count = len(string_sequences)
        if n_jobs != 1 and count != 1:
            pool = Pool(n_jobs)
            jobs = [
                pool.apply_async(
                    inv_transform,
                    args=(
                        string_sequences[i],
                        self.parameters.centers,
                        self.parameters.alphabets.tolist(),
                        start_set[i],
                        self.target_lengths_[i] if self.target_lengths_ is not None and i < len(self.target_lengths_) else None,
                    ),
                )
                for i in range(count)
            ]
            pool.close()
            pool.join()
            inverse_sequences = [job.get() for job in jobs]
        else:
            inverse_sequences = [
                inv_transform(
                    seq,
                    self.parameters.centers,
                    self.parameters.alphabets.tolist(),
                    start_set[i],
                    self.target_lengths_[i] if self.target_lengths_ is not None and i < len(self.target_lengths_) else None,
                )
                for i, seq in enumerate(string_sequences)
            ]
        if self.return_series_univariate:
            return np.hstack(inverse_sequences)
        return inverse_sequences

    def piece_to_symbol(self, piece):
        if self.parameters is None:
            raise ValueError("Please fit the model before transform.")
        piece = np.asarray(piece, dtype=np.float64)
        idx = int(np.argmin(np.linalg.norm(self.parameters.centers - piece, ord=2, axis=1)))
        return self.parameters.alphabets[idx]

    @staticmethod
    def string_separation(symbols, num_pieces):
        out = []
        csum = np.cumsum([0] + num_pieces)
        for i in range(len(num_pieces)):
            out.append(symbols[csum[i] : csum[i + 1]])
        return out

    @staticmethod
    def n_jobs_init(n_jobs=-1, _max=np.inf):
        if not isinstance(n_jobs, int):
            raise TypeError("Expected int type for n_jobs.")
        if n_jobs == 0:
            raise ValueError("Please feed a correct value for n_jobs.")
        if n_jobs in {None, -1}:
            n_jobs = _cpu_count()
        if n_jobs > _max:
            n_jobs = int(_max)
            warnings.warn(f"n_jobs is set to maximum {n_jobs}.")
        return max(int(n_jobs), 1)


def one_D_centers(data, labels, k):
    centers = np.zeros(k)
    for clust in np.unique(labels):
        centers[int(clust)] = np.mean(data[labels == clust])
    return centers


class fastABBA(object):
    """Small compatibility wrapper matching the original fastABBA entry point."""

    def __init__(self, tol=0.2, k=2, r=0.5, scl=1, random_state=2022, n_jobs=1, alphabet_set=0, max_len=np.inf, max_iter=2, verbose=True):
        self.model = JABBA(
            tol=tol,
            k=k,
            r=r,
            scl=scl,
            random_state=random_state,
            max_len=max_len,
            max_iter=max_iter,
            verbose=verbose,
            init="kmeans",
        )
        self.n_jobs = n_jobs
        self.alphabet_set = alphabet_set

    def transform(self, series):
        return self.model.fit_transform(series, n_jobs=self.n_jobs, alphabet_set=self.alphabet_set)[0]

    def predict(self, series):
        return self.model.transform_single_series(np.asarray(series, dtype=np.float64))

    def inverse_transform(self, strings, start):
        return inv_transform(strings, self.model.parameters.centers, self.model.parameters.alphabets.tolist(), start)


def general_compress(pabba, data, adjust=True, n_jobs=-1):
    data = np.asarray(data)
    if len(data.shape) > 3:
        raise ValueError("Please transform the shape of data into 1D, 2D or 3D.")
    if len(data.shape) == 3:
        pabba.d_shape = data.shape
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    else:
        pabba.d_shape = data.shape

    if adjust:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std == 0, 1, std)
        pabba.d_norm = (mean, std)
        return pabba.fit_transform((data - mean) / std, n_jobs=n_jobs)
    pabba.d_norm = None
    return pabba.fit_transform(data, n_jobs=n_jobs)


def general_decompress(pabba, strings, int_type=True, n_jobs=-1):
    reconstruction = np.asarray(pabba.inverse_transform(strings, n_jobs=n_jobs))
    if pabba.d_norm is not None:
        reconstruction = reconstruction * pabba.d_norm[1] + pabba.d_norm[0]
    if len(pabba.d_shape) == 3:
        reconstruction = reconstruction.reshape(pabba.d_shape)
        if int_type:
            reconstruction = reconstruction.round().astype(np.uint8)
    return reconstruction


def dtw(x, y, *, dist=lambda a, b: (a - b) * (a - b), return_path=False, filter_redundant=False):
    x = np.asarray(x)
    y = np.asarray(y)
    if filter_redundant:
        if len(x) > 2:
            xdiff = np.diff(x)
            x_keep = np.abs(xdiff[1:] - xdiff[:-1]) >= 1e-14
            x = x[np.hstack((True, x_keep, True))]
        else:
            x_keep = []
        if len(y) > 2:
            ydiff = np.diff(y)
            y_keep = np.abs(ydiff[1:] - ydiff[:-1]) >= 1e-14
            y = y[np.hstack((True, y_keep, True))]
        else:
            y_keep = []

    len_x, len_y = len(x), len(y)
    window = [(i + 1, j + 1) for i in range(len_x) for j in range(len_y)]
    D = defaultdict(lambda: (float("inf"),) if return_path else float("inf"))

    if return_path:
        if filter_redundant:
            x_ind = np.arange(1, len(x_keep) + 1)
            y_ind = np.arange(1, len(y_keep) + 1)
            x_ind = np.hstack((0, x_ind[x_keep], len(x_keep) + 1))
            y_ind = np.hstack((0, y_ind[y_keep], len(y_keep) + 1))
        else:
            x_ind = np.arange(len(x))
            y_ind = np.arange(len(y))
        D[0, 0] = (0, 0, 0)
        for i, j in window:
            d = dist(x[i - 1], y[j - 1])
            D[i, j] = min((D[i - 1, j][0] + d, i - 1, j), (D[i, j - 1][0] + d, i, j - 1), (D[i - 1, j - 1][0] + d, i - 1, j - 1), key=lambda a: a[0])
        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((x_ind[i - 1], y_ind[j - 1]))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return D[len_x, len_y][0], path

    D[0, 0] = 0
    for i, j in window:
        d = dist(x[i - 1], y[j - 1])
        D[i, j] = min(D[i - 1, j] + d, D[i, j - 1] + d, D[i - 1, j - 1] + d)
    return D[len_x, len_y]
