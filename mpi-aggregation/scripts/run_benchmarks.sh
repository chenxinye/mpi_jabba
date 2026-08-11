#!/usr/bin/env bash
set -euo pipefail

N=1000000
ALPHA=0.05
RANKS="1,2,4,8"
REPEAT=5
OUT="results/bench.csv"
DATASET="blobs"
SORTING="2-norm"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --sorting) SORTING="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$OUT")"
IFS=',' read -ra ARR <<< "$RANKS"
for p in "${ARR[@]}"; do
  mpirun -np "$p" ./build/apps/alphaagg_bench \
    --n "$N" --alpha "$ALPHA" --sorting "$SORTING" --dataset "$DATASET" \
    --seed "$SEED" --repeat "$REPEAT" --csv "$OUT"
done
