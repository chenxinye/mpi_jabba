from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    g = (
        df.groupby("rank_count", as_index=False)["wall_time_sec"]
        .median()
        .sort_values("rank_count")
        .rename(columns={"wall_time_sec": "median_time"})
    )
    t1 = float(g.loc[g["rank_count"] == 1, "median_time"].iloc[0])
    g["speedup"] = t1 / g["median_time"]
    g["efficiency"] = g["speedup"] / g["rank_count"]
    g.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    for y, fn, title, ylabel in [
        ("median_time", "runtime.png", "Runtime vs Rank Count", "Median wall time (s)"),
        ("speedup", "speedup.png", "Speedup vs Rank Count", "Speedup"),
        ("efficiency", "efficiency.png", "Parallel Efficiency vs Rank Count", "Efficiency"),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(g["rank_count"], g[y], marker="o", linewidth=2)
        plt.xlabel("Rank count")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fn), dpi=220)
        plt.close()


if __name__ == "__main__":
    main()
