#ifndef MPI_ALPHAAGG_ALPHAAGG_H
#define MPI_ALPHAAGG_ALPHAAGG_H

#include "mpi_alphaagg/alphaagg_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int alphaagg_serial_aggregate_2d(const double *points,
                                 int n_points,
                                 double alpha,
                                 alphaagg_sort_t sorting,
                                 alphaagg_result_t *out);

#ifdef __cplusplus
}
#endif

#endif
