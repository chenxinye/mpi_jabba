"""Revised JABBA package with MPI-backed aggregation."""

from .aggregation import aggregate_cython, aggregate_points, aggregate_serial_python
from .compression import compress, fillna
from .inverse import inv_compress, inv_digitize, inv_transform, quantize
from .jabba import JABBA, Model, dtw, fastABBA, general_compress, general_decompress, symbolsAssign

__version__ = "0.1.0"

__all__ = [
    "JABBA",
    "Model",
    "fastABBA",
    "compress",
    "fillna",
    "aggregate_points",
    "aggregate_cython",
    "aggregate_serial_python",
    "inv_transform",
    "inv_digitize",
    "inv_compress",
    "quantize",
    "symbolsAssign",
    "general_compress",
    "general_decompress",
    "dtw",
]
