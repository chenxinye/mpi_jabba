#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "mpi_alphaagg/mpi_alphaagg.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 || size > 4) {
        if (rank == 0) fprintf(stderr, "run with np=2 or 4\n");
        MPI_Finalize();
        return 1;
    }

    double chain_local[4];
    chain_local[0] = rank * 0.9;
    chain_local[1] = 0.0;
    chain_local[2] = rank * 0.9 + 0.9;
    chain_local[3] = 0.0;

    alphaagg_result_t res = {0};
    int rc = alphaagg_mpi_aggregate_2d(chain_local, 2, 1.0, ALPHAAGG_SORT_LEXI, MPI_COMM_WORLD, &res);
    assert(rc == 0);

    int local_ok = 1;
    for (int i = 1; i < res.n_points; ++i) {
        if (res.labels[i] != res.labels[0]) local_ok = 0;
    }
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    assert(all_ok == 1);
    assert(res.n_clusters == 1);
    alphaagg_result_free(&res);

    double far_local[4] = {rank * 10.0, 0.0, rank * 10.0 + 5.0, 0.0};
    rc = alphaagg_mpi_aggregate_2d(far_local, 2, 1.0, ALPHAAGG_SORT_2_NORM, MPI_COMM_WORLD, &res);
    assert(rc == 0);
    assert(res.n_clusters >= size);
    alphaagg_result_free(&res);

    double ptga_local[4];
    ptga_local[0] = (rank % 2 == 0) ? 0.0 : 0.04;
    ptga_local[1] = 0.0;
    ptga_local[2] = 10.0 + (double)rank;
    ptga_local[3] = 0.0;
    rc = alphaagg_mpi_ptga_aggregate_2d(ptga_local, 2, 0.1, ALPHAAGG_SORT_LEXI, MPI_COMM_WORLD, &res);
    assert(rc == 0);
    assert(res.labels[0] == 0);
    assert(res.n_clusters >= 2);
    alphaagg_result_free(&res);

    if (rank == 0) printf("test_mpi_small ok\n");
    MPI_Finalize();
    return 0;
}
