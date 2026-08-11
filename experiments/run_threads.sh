#!/usr/bin/env bash
set -euo pipefail

# JABBA worker/partition scaling.
#
# Default EXPERIMENT=pipeline reproduces the old mpi_jabba/multithreading.py
# protocol with experiments/jabba: full fit_transform + inverse_transform,
# partition=n_jobs, and legacy inverse timing. Set EXPERIMENT=compression to run
# the newer compression-only diagnostic.
#
CALLER_PWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-  --map-by core --bind-to core}"
EXPERIMENT="${EXPERIMENT:-pipeline}"
USER_OUTDIR="${OUTDIR:-}"
if [[ -n "${USER_OUTDIR}" ]]; then
    OUTDIR="${USER_OUTDIR}"
    if [[ "${OUTDIR}" != /* ]]; then
        OUTDIR="${CALLER_PWD}/${OUTDIR}"
    fi
elif [[ "${EXPERIMENT}" == "pipeline" ]]; then
    OUTDIR="${REPO_ROOT}/experiments/results_mthread_univariate"
else
    OUTDIR="${REPO_ROOT}/experiments/results_compression_threads"
fi
THREAD_VALUES="${THREAD_VALUES:-1 2 4 8 16 32}"
N_VALUES="${N_VALUES:-200000}"
TOL_VALUES="${TOL_VALUES:-0.01 0.005}"
TOL_VALUE="${TOL:-${TOL_VALUES%% *}}"
ALPHA="${ALPHA:-0.05}"
KIND="${KIND:-gaussian}"
SEED="${SEED:-0}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-3}"
COMPRESSION_BACKEND="${COMPRESSION_BACKEND:-cython}"
INVERSE_BACKEND="${INVERSE_BACKEND:-legacy-python}"
SPEEDUP_METRIC="${SPEEDUP_METRIC:-total}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-mpi}"
RESET="${RESET:-1}"
ALLOW_OVERSUBSCRIBE="${ALLOW_OVERSUBSCRIBE:-0}"
OVERSUBSCRIBE_ARG=""
if [[ "${ALLOW_OVERSUBSCRIBE}" == "1" || "${ALLOW_OVERSUBSCRIBE}" == "true" || "${ALLOW_OVERSUBSCRIBE}" == "TRUE" || "${ALLOW_OVERSUBSCRIBE}" == "yes" ]]; then
    OVERSUBSCRIBE_ARG="--allow-oversubscribe"
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "Table 2 compression thread/partition scaling"
echo "Repository : ${REPO_ROOT}"
echo "Output dir : ${OUTDIR}"
echo "Python     : ${PYTHON_BIN}"
echo "Experiment : ${EXPERIMENT}"
echo "Execution  : ${EXECUTION_BACKEND}"
echo "Workers    : ${THREAD_VALUES}"
echo "N values   : ${N_VALUES}"
echo "Tol        : ${TOL_VALUE}"
echo "Alpha      : ${ALPHA}"
echo "Backend    : ${COMPRESSION_BACKEND}"
echo "Inverse    : ${INVERSE_BACKEND}"
echo "Speedup    : ${SPEEDUP_METRIC}"
echo "Repeats    : ${REPEATS} (warmup=${WARMUP})"
echo "Oversub.   : ${ALLOW_OVERSUBSCRIBE}"
echo "============================================================"

if [[ "${EXPERIMENT}" == "pipeline" ]]; then
    "${PYTHON_BIN}" -u experiments/multithreading.py \
        --lengths ${N_VALUES} \
        --threads ${THREAD_VALUES} \
        ${OVERSUBSCRIBE_ARG:+${OVERSUBSCRIBE_ARG}} \
        --tol "${TOL_VALUE}" \
        --alpha "${ALPHA}" \
        --seed "${SEED}" \
        --kind "${KIND}" \
        --warmup "${WARMUP}" \
        --repeats "${REPEATS}" \
        --baseline-warmup 0 \
        --baseline-repeats 1 \
        --jabba-compression-backend "${COMPRESSION_BACKEND}" \
        --jabba-inverse-backend "${INVERSE_BACKEND}" \
        --jabba-speedup-metric "${SPEEDUP_METRIC}" \
        --mpi-processes 1 \
        --outdir "${OUTDIR}" \
        --formats pdf png
elif [[ "${EXECUTION_BACKEND}" == "mpi" ]]; then
    first_run=1
    for workers in ${THREAD_VALUES}; do
        write_mode=(--append)
        if [[ "${first_run}" == "1" && "${RESET}" == "1" ]]; then
            write_mode=(--reset)
        fi
        first_run=0
        echo
        echo "Launching MPI compression run with workers=partitions=${workers}"
        "${MPIRUN_BIN}" ${MPI_EXTRA_ARGS} -np "${workers}" "${PYTHON_BIN}" -u experiments/synth_compression.py \
            --n ${N_VALUES} \
            --tol ${TOL_VALUES} \
            --threads "${workers}" \
            --kind "${KIND}" \
            --seed "${SEED}" \
            --warmup "${WARMUP}" \
            --repeats "${REPEATS}" \
            --compression-backend "${COMPRESSION_BACKEND}" \
            --execution-backend mpi \
            --outdir "${OUTDIR}" \
            --csv-name compression_threads.csv \
            "${write_mode[@]}"
    done
else
    reset_arg=()
    if [[ "${RESET}" == "1" ]]; then
        reset_arg=(--reset)
    fi
    "${PYTHON_BIN}" -u experiments/synth_compression.py \
        --n ${N_VALUES} \
        --tol ${TOL_VALUES} \
        --threads ${THREAD_VALUES} \
        --kind "${KIND}" \
        --seed "${SEED}" \
        --warmup "${WARMUP}" \
        --repeats "${REPEATS}" \
        --compression-backend "${COMPRESSION_BACKEND}" \
        --execution-backend thread \
        --outdir "${OUTDIR}" \
        --csv-name compression_threads.csv \
        "${reset_arg[@]}"
fi

echo
echo "DONE: thread/partition results are in ${OUTDIR}"
