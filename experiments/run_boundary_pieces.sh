#!/usr/bin/env bash
set -euo pipefail

LENGTHS="${LENGTHS:-20000}"
PARTITIONS="${PARTITIONS:-1 2 4 8 16 32}"
TOL="${TOL:-0.5}"
KIND="${KIND:-random_walk}"
SEED="${SEED:-0}"
WARMUP="${WARMUP:-1}"
REPEATS="${REPEATS:-3}"
COMPRESSION_BACKEND="${COMPRESSION_BACKEND:-cython}"
OUTDIR="${OUTDIR:-results_boundary_pieces}"
FORMATS="${FORMATS:-pdf png}"

python compression_boundary_pieces.py \
  --lengths ${LENGTHS} \
  --partitions ${PARTITIONS} \
  --tol "${TOL}" \
  --kind "${KIND}" \
  --seed "${SEED}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --compression-backend "${COMPRESSION_BACKEND}" \
  --outdir "${OUTDIR}" \
  --formats ${FORMATS}
