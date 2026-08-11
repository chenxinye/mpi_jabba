from ._core import aggregate_mpi, aggregate_mpi_grid, aggregate_mpi_ptga, aggregate_serial
from .cython import aggregate_cython

__all__ = [
    "aggregate_serial",
    "aggregate_mpi",
    "aggregate_mpi_grid",
    "aggregate_mpi_ptga",
    "aggregate_cython",
    "__version__",
]
__version__ = "0.1.0"
