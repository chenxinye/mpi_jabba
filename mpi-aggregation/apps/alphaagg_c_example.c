#include <mpi.h>
#include <stdio.h>

#include "mpi_alphaagg/alphaagg.h"
#include "mpi_alphaagg/mpi_alphaagg.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double serial_points[] = {0.0, 0.0, 0.1, 0.1, 1.5, 1.5};
    alphaagg_result_t sres = {0};
    if (rank == 0) {
        alphaagg_serial_aggregate_2d(serial_points, 3, 0.3, ALPHAAGG_SORT_2_NORM, &sres);
        printf("serial n_clusters=%d total_sse=%.6f\n", sres.n_clusters, sres.total_sse);
        alphaagg_result_free(&sres);
    }

    double local_points[4] = {(double)rank * 0.9, 0.0, (double)rank * 0.9 + 0.2, 0.0};
    alphaagg_result_t mres = {0};
    alphaagg_mpi_aggregate_2d(local_points, 2, 1.0, ALPHAAGG_SORT_LEXI, MPI_COMM_WORLD, &mres);
    printf("rank=%d local_labels=[%d,%d] global_clusters=%d\n",
           rank,
           mres.n_points > 0 ? mres.labels[0] : -1,
           mres.n_points > 1 ? mres.labels[1] : -1,
           mres.n_clusters);
    alphaagg_result_free(&mres);

    MPI_Finalize();
    return 0;
}
