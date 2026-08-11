# Algorithm

## PTGA path

`alphaagg_mpi_ptga_aggregate_2d` implements the parallel two-stage greedy aggregation described in the paper revision:

1. Each MPI rank runs the serial greedy aggregation on its local block.
2. Each rank sends only the local cluster starting points to rank 0.
3. Rank 0 runs the same greedy aggregation on those starting points.
4. Rank 0 scatters the local-cluster-to-global-cluster labels back to ranks.
5. Each rank propagates labels from local points to global labels and participates in reductions for cluster counts, centroids, and SSE.

This path avoids point-level all-to-all exchange. Its dominant communication is proportional to the number of local starting points, plus the final per-cluster reductions.

For the closest match to the algorithm in the paper, callers should distribute contiguous blocks of globally sorted points to ranks before calling PTGA.

## Grid/DSU path

`alphaagg_mpi_aggregate_2d` is retained as a stricter connectivity-oriented variant. It performs local serial micro-clustering, alpha-grid based point record exchange with `MPI_Alltoallv`, point-neighbor edge generation, centralized union-find on rank 0, broadcast of global labels, and global SSE summary reduction.

The grid/DSU path can merge cross-rank neighbor chains that PTGA intentionally approximates, but it communicates point records rather than only starting points and does not match the low-communication method described in the revision.
