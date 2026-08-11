#include "mpi_alphaagg/alphaagg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int index;
    double x;
    double y;
    double key_primary;
    double key_secondary;
} alphaagg_ordered_point_t;

static alphaagg_sort_t g_sorting;

static int alphaagg_cmp_point(const void *a, const void *b) {
    const alphaagg_ordered_point_t *pa = (const alphaagg_ordered_point_t *)a;
    const alphaagg_ordered_point_t *pb = (const alphaagg_ordered_point_t *)b;
    if (pa->key_primary < pb->key_primary) {
        return -1;
    }
    if (pa->key_primary > pb->key_primary) {
        return 1;
    }
    if (g_sorting == ALPHAAGG_SORT_LEXI) {
        if (pa->key_secondary < pb->key_secondary) {
            return -1;
        }
        if (pa->key_secondary > pb->key_secondary) {
            return 1;
        }
    }
    return (pa->index < pb->index) ? -1 : ((pa->index > pb->index) ? 1 : 0);
}

int alphaagg_serial_aggregate_2d(const double *points,
                                 int n_points,
                                 double alpha,
                                 alphaagg_sort_t sorting,
                                 alphaagg_result_t *out) {
    if (!out || n_points < 0 || alpha <= 0.0 || sorting < ALPHAAGG_SORT_2_NORM || sorting > ALPHAAGG_SORT_LEXI) {
        return 1;
    }
    if (n_points > 0 && !points) {
        return 1;
    }

    out->n_points = 0;
    out->n_clusters = 0;
    out->labels = NULL;
    out->clusters = NULL;
    out->total_sse = 0.0;

    if (n_points == 0) {
        return 0;
    }

    alphaagg_ordered_point_t *order = (alphaagg_ordered_point_t *)malloc((size_t)n_points * sizeof(alphaagg_ordered_point_t));
    int *labels = (int *)malloc((size_t)n_points * sizeof(int));
    if (!order || !labels) {
        free(order);
        free(labels);
        return 2;
    }

    for (int i = 0; i < n_points; ++i) {
        double x = points[2 * i];
        double y = points[2 * i + 1];
        order[i].index = i;
        order[i].x = x;
        order[i].y = y;
        if (sorting == ALPHAAGG_SORT_2_NORM) {
            order[i].key_primary = sqrt(x * x + y * y);
            order[i].key_secondary = 0.0;
        } else if (sorting == ALPHAAGG_SORT_1_NORM) {
            order[i].key_primary = fabs(x) + fabs(y);
            order[i].key_secondary = 0.0;
        } else {
            order[i].key_primary = x;
            order[i].key_secondary = y;
        }
        labels[i] = -1;
    }

    g_sorting = sorting;
    qsort(order, (size_t)n_points, sizeof(alphaagg_ordered_point_t), alphaagg_cmp_point);

    int cluster_count = 0;
    double alpha2 = alpha * alpha;
    double *seed_buf_x = (double *)malloc((size_t)n_points * sizeof(double));
    double *seed_buf_y = (double *)malloc((size_t)n_points * sizeof(double));
    if (!seed_buf_x || !seed_buf_y) {
        free(order);
        free(labels);
        free(seed_buf_x);
        free(seed_buf_y);
        return 2;
    }

    for (int i = 0; i < n_points; ++i) {
        int seed_orig_idx = order[i].index;
        if (labels[seed_orig_idx] != -1) {
            continue;
        }
        int cid = cluster_count++;
        double seed_x = order[i].x;
        double seed_y = order[i].y;
        double seed_key = order[i].key_primary;
        seed_buf_x[cid] = order[i].x;
        seed_buf_y[cid] = order[i].y;
        labels[seed_orig_idx] = cid;

        for (int j = i + 1; j < n_points; ++j) {
            int candidate_orig_idx = order[j].index;
            if (labels[candidate_orig_idx] != -1) {
                continue;
            }
            double gap = order[j].key_primary - seed_key;
            if (gap > alpha) {
                break;
            }
            double dx = order[j].x - seed_x;
            double dy = order[j].y - seed_y;
            if (dx * dx + dy * dy <= alpha2) {
                labels[candidate_orig_idx] = cid;
            }
        }
    }

    alphaagg_cluster_t *clusters = (alphaagg_cluster_t *)calloc((size_t)cluster_count, sizeof(alphaagg_cluster_t));
    double *sum_x = (double *)calloc((size_t)cluster_count, sizeof(double));
    double *sum_y = (double *)calloc((size_t)cluster_count, sizeof(double));
    double *sum_sq = (double *)calloc((size_t)cluster_count, sizeof(double));
    if (!clusters || !sum_x || !sum_y || !sum_sq) {
        free(order);
        free(labels);
        free(seed_buf_x);
        free(seed_buf_y);
        free(clusters);
        free(sum_x);
        free(sum_y);
        free(sum_sq);
        return 2;
    }

    for (int i = 0; i < n_points; ++i) {
        int cid = labels[i];
        double x = points[2 * i];
        double y = points[2 * i + 1];
        clusters[cid].label = cid;
        clusters[cid].count += 1;
        sum_x[cid] += x;
        sum_y[cid] += y;
        sum_sq[cid] += x * x + y * y;
    }

    double total_sse = 0.0;
    for (int c = 0; c < cluster_count; ++c) {
        double inv = 1.0 / (double)clusters[c].count;
        double cx = sum_x[c] * inv;
        double cy = sum_y[c] * inv;
        double sse = sum_sq[c] - (double)clusters[c].count * (cx * cx + cy * cy);
        if (sse < 0.0 && sse > -1e-12) {
            sse = 0.0;
        }
        clusters[c].seed_x = seed_buf_x[c];
        clusters[c].seed_y = seed_buf_y[c];
        clusters[c].centroid_x = cx;
        clusters[c].centroid_y = cy;
        clusters[c].sse = sse;
        total_sse += sse;
    }

    free(order);
    free(seed_buf_x);
    free(seed_buf_y);
    free(sum_x);
    free(sum_y);
    free(sum_sq);

    out->n_points = n_points;
    out->n_clusters = cluster_count;
    out->labels = labels;
    out->clusters = clusters;
    out->total_sse = total_sse;
    return 0;
}
