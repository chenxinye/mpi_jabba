#include "alphaagg_internal.h"

#include <math.h>

static unsigned long long alphaagg_mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

long long alphaagg_cell_owner(long long cell_x, long long cell_y, int comm_size) {
    unsigned long long ux = (unsigned long long)cell_x;
    unsigned long long uy = (unsigned long long)cell_y;
    unsigned long long h = alphaagg_mix64(ux * 0x9e3779b97f4a7c15ULL ^ (uy + 0x632be59bd9b4e019ULL));
    return (long long)(h % (unsigned long long)comm_size);
}

void alphaagg_point_to_cell(double x, double y, double alpha, long long *cell_x, long long *cell_y) {
    *cell_x = (long long)floor(x / alpha);
    *cell_y = (long long)floor(y / alpha);
}
