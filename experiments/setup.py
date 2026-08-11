from __future__ import annotations

from pathlib import Path

import numpy as np
from setuptools import Extension, find_packages, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Cython is required to build revised JABBA extensions") from exc

ROOT = Path(__file__).parent

extensions = [
    Extension("jabba.compmem", [str(ROOT / "jabba" / "compmem.pyx")], include_dirs=[np.get_include()]),
    Extension("jabba.legacy_compmem", [str(ROOT / "jabba" / "legacy_compmem.pyx")], include_dirs=[np.get_include()]),
    Extension("jabba.aggmem", [str(ROOT / "jabba" / "aggmem.pyx")], include_dirs=[np.get_include()]),
    Extension("jabba.inversetc", [str(ROOT / "jabba" / "inversetc.pyx")], include_dirs=[np.get_include()]),
]

setup(
    name="jabba",
    version="0.2.0",
    description="Revised JABBA with Cython kernels and MPI aggregation adapter",
    packages=find_packages(include=["jabba", "jabba.*"]),
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    install_requires=["numpy", "cython"],
    python_requires=">=3.9",
)
