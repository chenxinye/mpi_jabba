#include "alphaagg_internal.h"

#include <stdlib.h>

void alphaagg_result_clear(alphaagg_result_t *out) {
    if (!out) {
        return;
    }
    out->n_points = 0;
    out->n_clusters = 0;
    out->labels = NULL;
    out->clusters = NULL;
    out->total_sse = 0.0;
}

void alphaagg_result_free(alphaagg_result_t *res) {
    if (!res) {
        return;
    }
    free(res->labels);
    free(res->clusters);
    alphaagg_result_clear(res);
}
