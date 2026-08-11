# Revised JABBA

This package is a revised JABBA implementation for the paper revision. It keeps
JABBA's original multithreaded compression/reconstruction structure, adds the
high-performance Cython kernels from fABBA/JABBA, and routes `init="agg"`
digitization through the standalone `mpi-aggregation` implementation when that
backend is available.

The code is self-contained under `revision_new/jabba` and does not depend on the
older `software/qabba` package.

## What Is Included

Core Python API:

- `JABBA`
- `fastABBA`
- `compress`
- `aggregate_points`
- `inv_digitize`
- `inv_compress`
- `inv_transform`
- `general_compress`
- `general_decompress`
- `dtw`

Cython acceleration modules:

- `jabba.compmem`: compression
- `jabba.aggmem`: serial alpha-neighborhood aggregation
- `jabba.inversetc`: inverse digitization, quantization, inverse compression

Runtime fallback behavior:

- Compression uses `jabba.compmem.compress` when compiled, otherwise Python.
- Inverse digitization/compression uses `jabba.inversetc` when compiled, otherwise Python.
- Aggregation uses `mpi_alphaagg` when available. Under `mpirun`, it can use MPI
  PTGA/grid aggregation. Outside MPI, it uses serial `mpi_alphaagg` when
  available, then `jabba.aggmem`, then Python.

The selected aggregation backend is stored in `model.aggregation_backend_`.

## Build Requirements

```bash
python -m pip install numpy cython setuptools wheel
```

Optional helpers for compatibility with older JABBA workflows:

```bash
python -m pip install scikit-learn joblib
```

For MPI aggregation, build or install `mpi-aggregation` first from the repository
root:

```bash
cd mpi-aggregation
python -m pip install -v .
```

## Compile The Cython Kernels

From `revision_new`:

```bash
cd /Users/chenxinye/mpi_jabba/revision_new
python setup.py build_ext --inplace
```

Editable install:

```bash
python -m pip install -e .
```

If build isolation cannot find NumPy headers, use:

```bash
python -m pip install -e . --no-build-isolation
```

Quick import check:

```bash
PYTHONPATH=. python -c "from jabba import JABBA, compress, aggregate_cython; print(JABBA); print(compress([0.0, 1.0, 2.0], tol=0.1)); print(aggregate_cython([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0]], 0.05)['n_clusters'])"
```

## Basic Usage

```python
import numpy as np
from jabba import JABBA

x = np.linspace(0.0, 4.0 * np.pi, 400)
series = np.sin(x) + 0.2 * np.sin(5.0 * x)

model = JABBA(tol=0.05, alpha=0.25, sorting="2-norm")
strings, starts = model.fit_transform(series, n_jobs=2, return_start_set=True)
reconstruction = model.inverse_transform(strings, starts)

print(strings)
print(len(reconstruction))
print(model.aggregation_backend_)
```

## Multithreading

JABBA uses multithreading through `n_jobs` for compression and inverse
reconstruction. For a single long univariate series, `n_jobs > 1` partitions the
series into contiguous blocks and compresses each block in a thread pool. For a
2D input, each row is treated as an independent time series and rows are
processed concurrently.

```python
model = JABBA(tol=0.04, alpha=0.2)
strings, starts = model.fit_transform(series, n_jobs=4, return_start_set=True)
reconstruction = model.inverse_transform(strings, starts, n_jobs=4)
```

Use `n_jobs=1` when you want fully deterministic single-thread debugging. Use a
larger value when compression/reconstruction dominates runtime and the input has
enough points or enough independent rows to amortize thread overhead.

## MPI Multiprocessing

MPI is used for the aggregation step. Compression and inverse reconstruction are
local operations and still use the Cython kernels plus optional thread pools.

Run an example with multiple MPI ranks:

```bash
cd examples
mpirun -np 4 python simple_transform.py
```

Inside Python, keep the default `prefer_mpi=True` and choose the MPI aggregation
algorithm:

```python
from jabba import JABBA

model = JABBA(
    tol=0.04,
    alpha=0.2,
    sorting="2-norm",
    mpi_algorithm="ptga",
    prefer_mpi=True,
)
strings, starts = model.fit_transform(series, n_jobs=4, return_start_set=True)
```

Notes:

- `mpi_algorithm="ptga"` selects the low-communication MPI aggregation path.
- `mpi_algorithm="grid"` selects the grid/DSU aggregation path when supported by
  the installed `mpi_alphaagg` backend.
- Outside `mpirun`, the adapter intentionally avoids MPI collectives and uses
  serial aggregation to prevent MPI runtime networking/binding warnings.
- If `mpi_alphaagg` is unavailable, aggregation falls back to the local Cython
  serial aggregator, then to a pure Python reference implementation.

## Examples

The examples are intentionally simple function-based scripts and do not use
`argparse`.

```bash
cd jabba/examples
PYTHONPATH=. python simple_transform.py
PYTHONPATH=. python simple_components.py
```

`simple_transform.py` checks the public `JABBA.fit_transform` and
`JABBA.inverse_transform` flow. `simple_components.py` checks compression,
digitization, inverse digitization, inverse compression, and reconstruction as
separate pieces.

## Development Checks

```bash
cd jabba
PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile *.py examples/*.py
python setup.py build_ext --inplace
cd examples
PYTHONPATH=. python simple_components.py
PYTHONPATH=. python simple_transform.py
```

## Source Notes

The package keeps the public API close to the original JABBA/fABBA interface,
including `JABBA`, `fit_transform`, `transform`, `inverse_transform`, `compress`,
`digitize`, and the compatibility alias `digitizate`. The Cython modules are
organized around the same responsibilities as the fABBA/JABBA implementation:
compression, aggregation/digitization support, inverse digitization, and inverse
compression.
