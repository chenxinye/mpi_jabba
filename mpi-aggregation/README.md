# mpi-alpha-agg

MPI-based alpha-neighborhood aggregation for 2D data with a C core, MPI implementation, benchmarks, plots, and Python bindings.

## Features

- Serial greedy 2D alpha aggregation (`alphaagg_serial_aggregate_2d`)
- Low-communication PTGA MPI aggregation (`alphaagg_mpi_ptga_aggregate_2d`) matching the paper revision
- Grid/DSU MPI aggregation (`alphaagg_mpi_aggregate_2d`) with globally consistent near-neighbor labels
- Bundled fABBA Cython aggregation reference (`aggregate_cython`)
- PTGA communication over local starting points only, followed by label propagation
- Optional alpha-grid point exchange using `MPI_Alltoallv`
- C benchmark executable and Python benchmark frontend
- Plot generation for runtime, speedup, and efficiency

## Build (C/C++)

```bash
cmake -S . -B build     -DPython_EXECUTABLE="$(which python)"     -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## C usage

Include headers from `include/mpi_alphaagg` and link `alphaagg`.

- `alphaagg_serial_aggregate_2d(points, n, alpha, sorting, &out)`
- `alphaagg_mpi_ptga_aggregate_2d(local_points, local_n, alpha, sorting, MPI_COMM_WORLD, &out)`
- `alphaagg_mpi_aggregate_2d(local_points, local_n, alpha, sorting, MPI_COMM_WORLD, &out)` for the grid/DSU variant
- `alphaagg_result_free(&out)`

## Python install and usage

```bash
python -m pip install -v .
python -c "import mpi_alphaagg; print(mpi_alphaagg.__version__)"
```

```python
import numpy as np
import mpi_alphaagg as aa

x = np.random.rand(1000, 2)
out = aa.aggregate_serial(x, 0.05, "2-norm")
cython_out = aa.aggregate_cython(x, 0.05, "2-norm")
```

MPI Python binding (run under MPI launcher):

```bash
mpirun -np 4 python -m mpi_alphaagg.benchmark --algorithm ptga --n 100000 --alpha 0.05 --dataset blobs --csv results/py_bench.csv
```

## C benchmark

```bash
mpirun -np 4 ./build/apps/alphaagg_bench --algorithm ptga --n 100000 --alpha 0.05 --repeat 3 --csv results/bench.csv
```

Or run multiple ranks:

```bash
./scripts/run_benchmarks.sh --n 1000000 --alpha 0.05 --ranks 1,2,4,8 --repeat 5 --out results/bench.csv
```

## Plotting

```bash
python -m mpi_alphaagg.plot_benchmark results/bench.csv --outdir results/plots
```

Compare the native serial Python binding with the bundled Cython reference using the same
dataset, alpha, and sorting parameters:

```bash
python -m mpi_alphaagg.cython_benchmark --n 100000 --alpha 0.05 --dataset blobs --csv results/cython_bench.csv
```

Outputs:

- `runtime.png`
- `speedup.png`
- `efficiency.png`
- `summary.csv`

## MPI caveats

- MPI mode assumes launch under `mpirun`/`mpiexec`.
- PTGA is the paper algorithm: each rank aggregates its local block, rank 0 aggregates the resulting starting points, and ranks propagate global labels locally. Communication is proportional to the number of local clusters, not the number of original points.
- For the closest match to the paper, feed each rank a contiguous block of the globally sorted points.
- Different rank counts may produce different cluster partitions because PTGA relaxes global greedy ordering at block boundaries.
- The grid/DSU path provides a stricter near-neighbor connectivity merge, but exchanges point records with `MPI_Alltoallv` and is not the low-communication algorithm described in the revision.

## Packaging

Uses `scikit-build-core` + `pybind11`.

```bash
python -m build
python -m twine check dist/*
```

PyPI/binary users still need a compatible system MPI runtime.
