#ifndef MPI_ALPHAAGG_MPI_ALPHAAGG_H
#define MPI_ALPHAAGG_MPI_ALPHAAGG_H

#include <mpi.h>

#include "mpi_alphaagg/alphaagg_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int alphaagg_mpi_aggregate_2d(const double *local_points,
                              int local_n,
                              double alpha,
                              alphaagg_sort_t sorting,
                              MPI_Comm comm,
                              alphaagg_result_t *local_out);

int alphaagg_mpi_ptga_aggregate_2d(const double *local_points,
                                   int local_n,
                                   double alpha,
                                   alphaagg_sort_t sorting,
                                   MPI_Comm comm,
                                   alphaagg_result_t *local_out);

#ifdef __cplusplus
}
#endif

#endif
