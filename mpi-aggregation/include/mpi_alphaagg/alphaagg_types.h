#ifndef MPI_ALPHAAGG_TYPES_H
#define MPI_ALPHAAGG_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ALPHAAGG_SORT_2_NORM = 0,
    ALPHAAGG_SORT_1_NORM = 1,
    ALPHAAGG_SORT_LEXI = 2
} alphaagg_sort_t;

typedef struct {
    int label;
    int count;
    double seed_x;
    double seed_y;
    double centroid_x;
    double centroid_y;
    double sse;
} alphaagg_cluster_t;

typedef struct {
    int n_points;
    int n_clusters;
    int *labels;
    alphaagg_cluster_t *clusters;
    double total_sse;
} alphaagg_result_t;

void alphaagg_result_free(alphaagg_result_t *res);

#ifdef __cplusplus
}
#endif

#endif
