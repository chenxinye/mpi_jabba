#include <assert.h>
#include <stdio.h>

#include "mpi_alphaagg/alphaagg.h"

int main(void) {
    double pts[] = {0.0, 0.0, 0.1, 0.0, 2.0, 0.0};
    alphaagg_result_t res = {0};
    int rc = alphaagg_serial_aggregate_2d(pts, 3, 0.2, ALPHAAGG_SORT_2_NORM, &res);
    assert(rc == 0);
    assert(res.n_clusters == 2);
    alphaagg_result_free(&res);

    rc = alphaagg_serial_aggregate_2d(pts, 3, 5.0, ALPHAAGG_SORT_LEXI, &res);
    assert(rc == 0);
    assert(res.n_clusters == 1);
    alphaagg_result_free(&res);

    rc = alphaagg_serial_aggregate_2d(pts, 3, 0.2, ALPHAAGG_SORT_1_NORM, &res);
    assert(rc == 0);
    alphaagg_result_free(&res);

    printf("test_serial ok\n");
    return 0;
}
