# Packaging

This project uses CMake for native builds and `scikit-build-core` for Python wheels.

Build wheel:

```bash
python -m build
python -m twine check dist/*
```

MPI-enabled wheels still rely on a valid system MPI runtime and development headers/libraries.
