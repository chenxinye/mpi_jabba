#ifndef ALPHAAGG_INTERNAL_H
#define ALPHAAGG_INTERNAL_H

#include <mpi.h>

#include "mpi_alphaagg/alphaagg_types.h"

typedef struct {
    long long parent;
    int rank;
} alphaagg_dsu_node_t;

typedef struct {
    long long global_point_id;
    int owner_rank;
    int local_index;
    long long micro_cluster_id;
    double x;
    double y;
    long long cell_x;
    long long cell_y;
} alphaagg_point_record_t;

typedef struct {
    long long a;
    long long b;
} alphaagg_edge_t;

int alphaagg_dsu_init(alphaagg_dsu_node_t **nodes, long long n);
long long alphaagg_dsu_find(alphaagg_dsu_node_t *nodes, long long x);
void alphaagg_dsu_union(alphaagg_dsu_node_t *nodes, long long a, long long b);

long long alphaagg_cell_owner(long long cell_x, long long cell_y, int comm_size);
void alphaagg_point_to_cell(double x, double y, double alpha, long long *cell_x, long long *cell_y);

void alphaagg_result_clear(alphaagg_result_t *out);

#endif
