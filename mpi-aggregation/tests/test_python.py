import os
import subprocess
import sys

import numpy as np


def test_import():
    import mpi_alphaagg as m

    assert m.__version__


def test_serial_binding():
    import mpi_alphaagg as m

    pts = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]], dtype=np.float64)
    out = m.aggregate_serial(pts, 0.2)
    assert int(out["n_clusters"]) == 2


def test_cython_binding_matches_serial_shape():
    import mpi_alphaagg as m

    pts = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 0.0]], dtype=np.float64)
    out = m.aggregate_cython(pts, 0.2)
    assert int(out["n_clusters"]) == 2
    assert out["labels"].shape == (3,)
    assert out["clusters"].shape == (2, 7)
    assert float(out["total_sse"]) >= 0.0


def test_mpi_binding_smoke():
    cmd = [
        "mpirun",
        "--oversubscribe",
        "-np",
        "2",
        sys.executable,
        "-c",
        (
            "import numpy as np, mpi_alphaagg as a; "
            "x=np.array([[0.0,0.0],[0.9,0.0]],dtype=np.float64); "
            "o=a.aggregate_mpi(x,1.0); "
            "print(o['n_clusters'])"
        ),
    ]
    env = os.environ.copy()
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_cython_benchmark_smoke(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "mpi_alphaagg.cython_benchmark",
        "--n",
        "20",
        "--alpha",
        "0.2",
        "--repeat",
        "1",
        "--csv",
        str(tmp_path / "cython_bench.csv"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
