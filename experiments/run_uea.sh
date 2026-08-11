#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:- --map-by core --bind-to core}"
M_VALUES="${M_VALUES:-2 4 8 16 32}"
DATA_DIR="${DATA_DIR:-../UEA2018}"
OUTDIR="${OUTDIR:-results_uea_total}"
DATASET="${DATASET:-all}"
REPEATS="${REPEATS:-3}"
ALPHA="${ALPHA:-0.01}"
SORTING="${SORTING:-2-norm}"
CENTER_KIND="${CENTER_KIND:-seed}"
SCL="${SCL:-1.0}"
COMPRESS_JOBS="${COMPRESS_JOBS:-0}"
INVERSE_JOBS="${INVERSE_JOBS:--1}"
RESET="${RESET:-1}"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "UEA MPI total wall-clock scaling"
echo "Directory     : ${SCRIPT_DIR}"
echo "Python        : ${PYTHON_BIN}"
echo "MPI runner    : ${MPIRUN_BIN}"
echo "M values      : ${M_VALUES}"
echo "MPI extra args: ${MPI_EXTRA_ARGS}"
echo "Dataset       : ${DATASET}"
echo "Data dir      : ${DATA_DIR}"
echo "Output dir    : ${OUTDIR}"
echo "Repeats       : ${REPEATS}"
echo "Compress jobs : ${COMPRESS_JOBS} (0 = one local worker per MPI rank)"
echo "Inverse jobs  : ${INVERSE_JOBS} (-1/0 = one local worker per MPI rank)"
echo "============================================================"

read -r -a M_ARRAY <<< "${M_VALUES}"
for idx in "${!M_ARRAY[@]}"; do
    M="${M_ARRAY[$idx]}"
    RESET_ARG=()
    if [[ "${RESET}" == "1" && "${idx}" == "0" ]]; then
        RESET_ARG=(--reset)
    fi

    echo
    echo "------------------------------------------------------------"
    echo "Running UEA total-runtime experiment with M=${M}"
    echo "Reset args: ${RESET_ARG[*]:-(none)}"
    echo "------------------------------------------------------------"

    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 "${MPIRUN_BIN}" ${MPI_EXTRA_ARGS} -np "${M}" \
        "${PYTHON_BIN}" -u uea_total_runtime.py \
        --dataset "${DATASET}" \
        --data-dir "${DATA_DIR}" \
        --outdir "${OUTDIR}" \
        --repeats "${REPEATS}" \
        --alpha "${ALPHA}" \
        --sorting "${SORTING}" \
        --center-kind "${CENTER_KIND}" \
        --scl "${SCL}" \
        --compress-jobs "${COMPRESS_JOBS}" \
        --inverse-jobs "${INVERSE_JOBS}" \
        "${RESET_ARG[@]}"
done

echo
echo "DONE: UEA results are in ${OUTDIR}"
