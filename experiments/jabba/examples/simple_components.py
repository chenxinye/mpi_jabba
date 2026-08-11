"""Small component-level example: compress, digitize, transform, reconstruct."""

from __future__ import annotations

import numpy as np

from jabba import JABBA, compress


def run_components():
    x = np.linspace(0.0, 4.0 * np.pi, 160)
    ts = np.sin(x)

    pieces = compress(ts, tol=0.03)
    model = JABBA(tol=0.03, alpha=0.2, verbose=0)
    strings = model.fit_transform(ts, n_jobs=1)
    transformed, starts = model.transform(ts, n_jobs=1)
    reconstructed = model.inverse_transform(strings, n_jobs=1)

    print("compressed pieces:", len(pieces))
    print("digitized symbols:", len(strings[0]))
    print("transform symbols:", len(transformed[0]))
    print("reconstruction length:", len(reconstructed))
    return pieces, strings, transformed, starts, reconstructed


if __name__ == "__main__":
    run_components()
