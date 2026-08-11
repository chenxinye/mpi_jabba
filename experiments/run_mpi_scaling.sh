#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root by default, even if launched from another folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"

# MPI extra arguments.
# For MPICH/Hydra, leave this empty.
# If using OpenMPI on another machine, you can run:
# MPI_EXTRA_ARGS="--oversubscribe" bash experiments/run_mpi_scaling.sh
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:- --map-by core --bind-to core}"

OUTDIR="${OUTDIR:-experiments/results_mpi_scaling}"
M_VALUES="${M_VALUES:-2 4 8 16 32}"
N_VALUES="${N_VALUES:-100000 500000}"
ALPHA_VALUES="${ALPHA_VALUES:-0.05 0.01}"
SORTING="${SORTING:-2-norm}"
SEED="${SEED:-0}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-3}"

mkdir -p "${OUTDIR}"

rm -f \
    "${OUTDIR}"/M*.csv \
    "${OUTDIR}"/rank*.json \
    "${OUTDIR}"/all_mpi_scaling.csv \
    "${OUTDIR}"/all_mpi_scaling.tex

# Turn the M list into a Bash array so we can show [current/total] progress.
read -r -a M_ARRAY <<< "${M_VALUES}"
TOTAL_M=${#M_ARRAY[@]}
CURRENT_M=0
ALL_START=$(date +%s)

echo "============================================================"
echo "Table 2 MPI scaling"
echo "Repository : ${REPO_ROOT}"
echo "Output dir : ${OUTDIR}"
echo "MPI runner : ${MPIRUN_BIN}"
echo "Python     : ${PYTHON_BIN}"
echo "M values   : ${M_VALUES}"
echo "N values   : ${N_VALUES}"
echo "Alpha      : ${ALPHA_VALUES}"
echo "Repeats    : ${REPEATS} (warmup=${WARMUP})"
echo "Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo

for M in "${M_ARRAY[@]}"; do
    CURRENT_M=$((CURRENT_M + 1))
    CSV_NAME="M${M}.csv"
    LOG_NAME="M${M}.log"
    RUN_START=$(date +%s)

    echo "------------------------------------------------------------"
    echo "[$CURRENT_M/$TOTAL_M] Starting M=${M} ranks"
    echo "Time       : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "CSV        : ${OUTDIR}/${CSV_NAME}"
    echo "Log        : ${OUTDIR}/${LOG_NAME}"
    echo "Parameters : n=[${N_VALUES}], alpha=[${ALPHA_VALUES}], repeats=${REPEATS}"
    echo "------------------------------------------------------------"

    # PYTHONUNBUFFERED=1 makes Python print output immediately instead of
    # keeping it in a buffer while mpirun is running.
    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 "${MPIRUN_BIN}" ${MPI_EXTRA_ARGS} -np "${M}" \
        "${PYTHON_BIN}" -u experiments/synth_aggregation.py \
        --n ${N_VALUES} \
        --alpha ${ALPHA_VALUES} \
        --sorting "${SORTING}" \
        --seed "${SEED}" \
        --warmup "${WARMUP}" \
        --repeats "${REPEATS}" \
        --outdir "${OUTDIR}" \
        --csv-name "${CSV_NAME}" 2>&1 | tee "${OUTDIR}/${LOG_NAME}"

    RUN_END=$(date +%s)
    RUN_SECONDS=$((RUN_END - RUN_START))

    if [[ -f "${OUTDIR}/${CSV_NAME}" ]]; then
        echo "[$CURRENT_M/$TOTAL_M] Finished M=${M} in ${RUN_SECONDS}s"
        echo "Saved      : ${OUTDIR}/${CSV_NAME}"
    else
        echo "[$CURRENT_M/$TOTAL_M] M=${M} finished, but CSV was not found: ${OUTDIR}/${CSV_NAME}" >&2
        exit 1
    fi
    echo
done

ALL_END=$(date +%s)
ALL_SECONDS=$((ALL_END - ALL_START))

echo "============================================================"
echo "All MPI runs finished in ${ALL_SECONDS}s"
echo "Combining CSV files and generating LaTeX..."
echo "============================================================"

"${PYTHON_BIN}" - "${OUTDIR}" <<'PY'
from pathlib import Path
import pandas as pd
import sys

outdir = Path(sys.argv[1])

files = sorted(
    outdir.glob("M*.csv"),
    key=lambda p: int(p.stem.split("M")[-1]),
)

if not files:
    raise SystemExit("no M*.csv files found")

frames = [pd.read_csv(path) for path in files]
df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["alpha", "n", "workers"])

combined = outdir / "all_mpi_scaling.csv"
df.to_csv(combined, index=False)

lines = [
    r"\begin{tabular}{c c c c c c c c c}",
    r"\toprule",
    r"$\alpha$ & $N$ & Method & $M$ & Time (s) & Speedup & SSE ratio & #Clusters (Serial) & #Clusters (MPI)\\",
    r"\midrule",
]

for (alpha, n), group in df.groupby(["alpha", "n"], sort=True):
    first = group.iloc[0]

    lines.append(
        f"{alpha:g} & {int(n)} & Serial GA & -- & "
        f"{first.serial_time_mean:.4f} & "
        f"1.000 & 1.0000 & {int(first.serial_clusters):,} & -- \\\\"
    )

    for _, row in group.iterrows():
        lines.append(
            f" & & MPI two-stage & {int(row.workers)} & "
            f"{row.mpi_time_mean:.4f} & "
            f"{row.speedup:.3f} & "
            f"{row.sse_ratio:.4f} & "
            f"{int(row.serial_clusters):,} & "
            f"{int(row.mpi_clusters):,} \\\\"
        )

    lines.append(r"\midrule")

if lines[-1] == r"\midrule":
    lines[-1] = r"\bottomrule"

lines.append(r"\end{tabular}")

tex = outdir / "all_mpi_scaling.tex"
tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Combined CSV: {combined}")
print(f"Combined LaTeX: {tex}")
PY

echo
echo "============================================================"
echo "DONE: all results are in ${OUTDIR}"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
