from __future__ import annotations

import argparse
import numpy as np

from . import aggregate_serial


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--sorting", default="2-norm")
    args = p.parse_args()

    pts = np.random.default_rng(42).random((args.n, 2))
    out = aggregate_serial(pts, args.alpha, args.sorting)
    print({"n_clusters": int(out["n_clusters"]), "total_sse": float(out["total_sse"])})


if __name__ == "__main__":
    main()
