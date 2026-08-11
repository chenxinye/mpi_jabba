#include "alphaagg_internal.h"

#include <stdlib.h>

int alphaagg_dsu_init(alphaagg_dsu_node_t **nodes, long long n) {
    if (!nodes || n < 0) {
        return 1;
    }
    *nodes = (alphaagg_dsu_node_t *)calloc((size_t)n, sizeof(alphaagg_dsu_node_t));
    if (!*nodes && n > 0) {
        return 2;
    }
    for (long long i = 0; i < n; ++i) {
        (*nodes)[i].parent = i;
        (*nodes)[i].rank = 0;
    }
    return 0;
}

long long alphaagg_dsu_find(alphaagg_dsu_node_t *nodes, long long x) {
    if (nodes[x].parent != x) {
        nodes[x].parent = alphaagg_dsu_find(nodes, nodes[x].parent);
    }
    return nodes[x].parent;
}

void alphaagg_dsu_union(alphaagg_dsu_node_t *nodes, long long a, long long b) {
    long long ra = alphaagg_dsu_find(nodes, a);
    long long rb = alphaagg_dsu_find(nodes, b);
    if (ra == rb) {
        return;
    }
    if (nodes[ra].rank < nodes[rb].rank) {
        nodes[ra].parent = rb;
    } else if (nodes[ra].rank > nodes[rb].rank) {
        nodes[rb].parent = ra;
    } else {
        nodes[rb].parent = ra;
        nodes[ra].rank += 1;
    }
}
