# Parallel Joint Symbolic Encoding of Time Series

This repository contains the experimental code for the paper **"Parallel Joint
Symbolic Encoding of Time Series"**  (previously accepted in [19th International Symposium on High-level Parallel Programming and Applications (HLPP)](https://sites.google.com/view/hlpp-2026/hlpp-2026)) by Xinye Chen, Sorbonne Université, CNRS,
LIP6.

The codebase focuses on a parallel implementation of JABBA-style symbolic time
series encoding. It combines Cython-accelerated compression and reconstruction
with an MPI/C implementation of alpha-neighborhood aggregation, enabling
experiments on synthetic data and benchmark time series datasets.

## Repository Structure

```text
.
├── experiments/          # Revised JABBA package and experiment scripts
│   ├── jabba/            # Python/Cython JABBA implementation
│   │   └── examples/     # Small usage examples
│   └── *.py, *.sh        # Experiment runners and plotting scripts
├── mpi-aggregation/      # C, MPI, and Python bindings for alpha aggregation
└── README.md             # Project overview
```

### Main Components

- **`experiments/jabba`**: revised JABBA implementation with Cython kernels for
  compression, aggregation, inverse digitization, and inverse reconstruction.
- **`mpi-aggregation`**: standalone MPI alpha-neighborhood aggregation library
  with C APIs, Python bindings, benchmarks, and tests.
- **Experiment scripts**: reproducibility scripts for synthetic aggregation,
  UEA digitization, multithreading, MPI scaling, and compression boundary
  analysis.

## Requirements

The exact environment may depend on the target machine and MPI installation.
The main requirements are:

- Python 3.9 or newer
- NumPy
- Cython
- setuptools and wheel
- CMake 3.20 or newer
- A C/C++ compiler
- MPI implementation such as Open MPI or MPICH
- pybind11

Optional packages used by some scripts include:

- scikit-learn
- joblib
- pandas
- matplotlib
- pytest
- build
- twine

## Installation

Clone the repository and enter the project root:

```bash
git clone <repository-url>
cd mpi_jabba
```

Install the Python build dependencies:

```bash
python -m pip install numpy cython setuptools wheel
```

Install optional development and plotting dependencies when needed:

```bash
python -m pip install scikit-learn joblib pandas matplotlib pytest build twine
```

### Build the MPI Aggregation Backend

The MPI backend is located in `mpi-aggregation`.

```bash
cd mpi-aggregation
python -m pip install -v .
```

Alternatively, build it with CMake:

```bash
cmake -S mpi-aggregation -B mpi-aggregation/build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build mpi-aggregation/build -j
ctest --test-dir mpi-aggregation/build --output-on-failure
```

### Build the Revised JABBA Package

From the `experiments` directory:

```bash
cd experiments
python setup.py build_ext --inplace
python -m pip install -e .
```

If build isolation cannot find NumPy headers, use:

```bash
python -m pip install -e . --no-build-isolation
```

## Quick Start

Run a small JABBA transform and inverse transform:

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

Run the bundled examples:

```bash
cd experiments/jabba/examples
PYTHONPATH=.. python simple_components.py
PYTHONPATH=.. python simple_transform.py
```

## Parallel Execution

### Multithreading

The revised JABBA implementation uses `n_jobs` for local parallel compression
and inverse reconstruction.

```python
model = JABBA(tol=0.04, alpha=0.2)
strings, starts = model.fit_transform(series, n_jobs=4, return_start_set=True)
reconstruction = model.inverse_transform(strings, starts, n_jobs=4)
```

For a long univariate series, the input is partitioned into contiguous blocks.
For a two-dimensional input, each row is treated as an independent time series.

### MPI Aggregation

MPI is used for the digitization aggregation step. Run MPI examples through
`mpirun` or `mpiexec`:

```bash
cd experiments/jabba/examples
mpirun -np 4 python simple_transform.py
```

The main MPI aggregation algorithms are:

- `ptga`: low-communication aggregation path used for the paper experiments.
- `grid`: grid/DSU aggregation path with stricter near-neighbor connectivity.

Outside an MPI launcher, the implementation avoids MPI collectives and falls
back to a serial aggregation backend.

## Reproducing Experiments

The experiment scripts are located in `experiments`.

### Synthetic Aggregation

```bash
cd experiments
mpirun -np 4 python synth_aggregation.py
```

### UEA Runtime and Digitization Experiments

Place the UEA archive in the expected directory layout:

```text
UEA2018/<Dataset>/<Dataset>_TRAIN.arff
UEA2018/<Dataset>/<Dataset>_TEST.arff
```

Then run the UEA scripts from `experiments`, for example:

```bash
mpirun -np 4 python uea_total_runtime.py --data-dir UEA2018
```

### Multithreading and MPI Scaling

Convenience shell scripts are included for repeated runs:

```bash
cd experiments
./run_threads.sh
./run_mpi_scaling.sh
./run_uea.sh
```

### Compression Boundary Analysis

```bash
cd experiments
./run_boundary_pieces.sh
```

Optional overrides:

```bash
LENGTHS="200000 500000" PARTITIONS="1 2 4 8 16 32" ./run_boundary_pieces.sh
```

## Datasets

The experiments use public time series classification archives:

- **UCR 2018 Time Series Classification Archive**:
  <https://www.cs.ucr.edu/~eamonn/time_series_data_2018/>
- **UEA 2018 Multivariate Time Series Classification Archive**:
  <https://www.timeseriesclassification.com/dataset.php>

Dataset files are not included in this repository. Download them separately and
place them under the paths expected by the experiment scripts.

## Development Checks

Useful checks for the JABBA package:

```bash
cd experiments
PYTHONPYCACHEPREFIX=/tmp/pycache python -m py_compile *.py jabba/*.py jabba/examples/*.py
python setup.py build_ext --inplace
cd jabba/examples
PYTHONPATH=.. python simple_components.py
PYTHONPATH=.. python simple_transform.py
```

Useful checks for the MPI aggregation package:

```bash
cmake -S mpi-aggregation -B mpi-aggregation/build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build mpi-aggregation/build -j
ctest --test-dir mpi-aggregation/build --output-on-failure
```

## Citation

If you use this repository in academic work, please cite the corresponding
paper. The formal publication metadata is not included in this repository yet;
please replace the placeholder fields below with the official venue, year, DOI,
or URL once available.

```bibtex
@misc{chen_parallel_joint_symbolic_encoding,
  title        = {Parallel Joint Symbolic Encoding of Time Series},
  author       = {Chen, Xinye},
  howpublished = {Manuscript in preparation},
  note         = {To appear; experimental code repository}
}
```

## Author

Xinye Chen  
Sorbonne Université, CNRS, LIP6  
Paris, France  
ORCID: <https://orcid.org/0000-0003-1778-393X>
