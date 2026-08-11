# Revision experiments

This folder contains clean reruns for the revised paper tables using the C/MPI implementation in `../mpi-aggregation`.

## Build prerequisite

From the repository root:

```bash
cmake -S mpi-aggregation -B mpi-aggregation/build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build mpi-aggregation/build -j
```

The scripts first try the installed `mpi_alphaagg` package, then fall back to `mpi-aggregation/build/_core*.so`.

## Table 2: synthetic aggregation

Run once per MPI size and append to the same CSV:

```bash
mpirun -np 4 python table2_synth_aggregation.py --outdir results_table2
mpirun -np 8 python table2_synth_aggregation.py --outdir results_table2
```

Outputs:

- `results_table2/table2_synth_aggregation.csv`
- `results_table2/table2_synth_aggregation.tex`
- `results_table2/table2_rank4.json`
- `results_table2/table2_rank8.json`

The script generates four Gaussian blobs, globally sorts points by 2-norm, scatters contiguous sorted blocks, runs PTGA, gathers local labels for SSE, and reports speedup as `serial_time / mpi_time`.

## Table 3: UEA digitization

The default UEA path matches the old script:

```text
UEA2018/<Dataset>/<Dataset>_TRAIN.arff
UEA2018/<Dataset>/<Dataset>_TEST.arff
```

Run once per MPI size:

```bash
mpirun -np 4 python table3_uea_digitization.py --data-dir UEA2018 --outdir results_table3
mpirun -np 8 python table3_uea_digitization.py --data-dir UEA2018 --outdir results_table3
```

Outputs:

- `results_table3/table3_uea_digitization.csv`
- `results_table3/table3_uea_digitization.tex`
- per-dataset JSON files such as `BasicMotions_K4.json`

This script reuses QABBA compression and inverse reconstruction, but replaces the digitization aggregation step with `mpi_alphaagg.aggregate_mpi_ptga`.

## Compression boundary piece-count metric

Run the compression-only boundary experiment:

```bash
./run_boundary_pieces.sh
```

Optional overrides:

```bash
LENGTHS="200000 500000" PARTITIONS="1 2 4 8 16 32" ./run_boundary_pieces.sh
```

Outputs:

- `results_boundary_pieces/compression_boundary_pieces.csv`
- `results_boundary_pieces/compression_boundary_pieces.tex`
- `results_boundary_pieces/compression_boundary_pieces.pdf`
- `results_boundary_pieces/compression_boundary_pieces.png`

The experiment compares serial compression with domain-decomposed compression
and reports the boundary-forced additional number of pieces, the relative
increase, and the theoretical boundary bound `M-1`. The CSV also includes the
observed piece-count change after locally recompressing each partition as a
diagnostic quantity.
