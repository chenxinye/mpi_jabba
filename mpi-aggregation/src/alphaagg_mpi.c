#include "mpi_alphaagg/mpi_alphaagg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "alphaagg_internal.h"
#include "mpi_alphaagg/alphaagg.h"

typedef struct {
    double x;
    double y;
    int local_cluster_id;
} alphaagg_seed_record_t;

static int add_edge(alphaagg_edge_t **edges, int *count, int *cap, long long a, long long b) {
    if (a == b) {
        return 0;
    }
    if (a > b) {
        long long t = a;
        a = b;
        b = t;
    }
    if (*count >= *cap) {
        int ncap = (*cap == 0) ? 256 : (*cap * 2);
        alphaagg_edge_t *tmp = (alphaagg_edge_t *)realloc(*edges, (size_t)ncap * sizeof(alphaagg_edge_t));
        if (!tmp) {
            return 2;
        }
        *edges = tmp;
        *cap = ncap;
    }
    (*edges)[*count].a = a;
    (*edges)[*count].b = b;
    *count += 1;
    return 0;
}

static int unique_neighbor_owners(long long cx, long long cy, int size, int owners[9]) {
    int n = 0;
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int o = (int)alphaagg_cell_owner(cx + (long long)dx, cy + (long long)dy, size);
            int seen = 0;
            for (int i = 0; i < n; ++i) {
                if (owners[i] == o) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) {
                owners[n++] = o;
            }
        }
    }
    return n;
}

int alphaagg_mpi_aggregate_2d(const double *local_points,
                              int local_n,
                              double alpha,
                              alphaagg_sort_t sorting,
                              MPI_Comm comm,
                              alphaagg_result_t *local_out) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!local_out || local_n < 0 || alpha <= 0.0 || sorting < ALPHAAGG_SORT_2_NORM || sorting > ALPHAAGG_SORT_LEXI) {
        return 1;
    }
    if (local_n > 0 && !local_points) {
        return 1;
    }
    alphaagg_result_clear(local_out);

    alphaagg_result_t micro = {0};
    int rc = alphaagg_serial_aggregate_2d(local_points, local_n, alpha, sorting, &micro);
    if (rc != 0) {
        return rc;
    }

    long long local_micro = (long long)micro.n_clusters;
    long long local_points_ll = (long long)local_n;
    long long micro_offset = 0;
    long long point_offset = 0;
    MPI_Exscan(&local_micro, &micro_offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Exscan(&local_points_ll, &point_offset, 1, MPI_LONG_LONG, MPI_SUM, comm);
    if (rank == 0) {
        micro_offset = 0;
        point_offset = 0;
    }

    long long total_micro = 0;
    MPI_Allreduce(&local_micro, &total_micro, 1, MPI_LONG_LONG, MPI_SUM, comm);

    int *send_counts = (int *)calloc((size_t)size, sizeof(int));
    int *send_displs = (int *)calloc((size_t)size, sizeof(int));
    int *recv_counts = (int *)calloc((size_t)size, sizeof(int));
    int *recv_displs = (int *)calloc((size_t)size, sizeof(int));
    long long *cell_x = NULL;
    long long *cell_y = NULL;
    if (!send_counts || !send_displs || !recv_counts || !recv_displs) {
        free(send_counts);
        free(send_displs);
        free(recv_counts);
        free(recv_displs);
        alphaagg_result_free(&micro);
        return 2;
    }

    if (local_n > 0) {
        cell_x = (long long *)malloc((size_t)local_n * sizeof(long long));
        cell_y = (long long *)malloc((size_t)local_n * sizeof(long long));
        if (!cell_x || !cell_y) {
            free(send_counts);
            free(send_displs);
            free(recv_counts);
            free(recv_displs);
            free(cell_x);
            free(cell_y);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    int send_total = 0;
    for (int i = 0; i < local_n; ++i) {
        alphaagg_point_to_cell(local_points[2 * i], local_points[2 * i + 1], alpha, &cell_x[i], &cell_y[i]);
        int owners[9];
        int no = unique_neighbor_owners(cell_x[i], cell_y[i], size, owners);
        send_total += no;
        for (int k = 0; k < no; ++k) {
            send_counts[owners[k]] += 1;
        }
    }

    for (int r = 1; r < size; ++r) {
        send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    }

    alphaagg_point_record_t *sendbuf = NULL;
    if (send_total > 0) {
        sendbuf = (alphaagg_point_record_t *)malloc((size_t)send_total * sizeof(alphaagg_point_record_t));
        if (!sendbuf) {
            free(send_counts);
            free(send_displs);
            free(recv_counts);
            free(recv_displs);
            free(cell_x);
            free(cell_y);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    int *cursor = (int *)malloc((size_t)size * sizeof(int));
    if (!cursor && size > 0) {
        free(send_counts);
        free(send_displs);
        free(recv_counts);
        free(recv_displs);
        free(sendbuf);
        free(cell_x);
        free(cell_y);
        alphaagg_result_free(&micro);
        return 2;
    }
    for (int r = 0; r < size; ++r) {
        cursor[r] = send_displs[r];
    }

    for (int i = 0; i < local_n; ++i) {
        int owners[9];
        int no = unique_neighbor_owners(cell_x[i], cell_y[i], size, owners);
        for (int k = 0; k < no; ++k) {
            int o = owners[k];
            int pos = cursor[o]++;
            sendbuf[pos].global_point_id = point_offset + (long long)i;
            sendbuf[pos].owner_rank = rank;
            sendbuf[pos].local_index = i;
            sendbuf[pos].micro_cluster_id = micro_offset + (long long)micro.labels[i];
            sendbuf[pos].x = local_points[2 * i];
            sendbuf[pos].y = local_points[2 * i + 1];
            sendbuf[pos].cell_x = cell_x[i];
            sendbuf[pos].cell_y = cell_y[i];
        }
    }

    free(cursor);
    free(cell_x);
    free(cell_y);

    MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
    for (int r = 1; r < size; ++r) {
        recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }

    int recv_total = 0;
    for (int r = 0; r < size; ++r) {
        recv_total += recv_counts[r];
    }

    alphaagg_point_record_t *recvbuf = NULL;
    if (recv_total > 0) {
        recvbuf = (alphaagg_point_record_t *)malloc((size_t)recv_total * sizeof(alphaagg_point_record_t));
        if (!recvbuf) {
            free(send_counts);
            free(send_displs);
            free(recv_counts);
            free(recv_displs);
            free(sendbuf);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    MPI_Datatype point_type;
    MPI_Type_contiguous((int)sizeof(alphaagg_point_record_t), MPI_BYTE, &point_type);
    MPI_Type_commit(&point_type);
    MPI_Alltoallv(sendbuf, send_counts, send_displs, point_type, recvbuf, recv_counts, recv_displs, point_type, comm);
    MPI_Type_free(&point_type);

    free(send_counts);
    free(send_displs);
    free(recv_counts);
    free(recv_displs);
    free(sendbuf);

    alphaagg_edge_t *edges = NULL;
    int n_edges = 0;
    int edges_cap = 0;
    const double a2 = alpha * alpha;

    for (int i = 0; i < recv_total; ++i) {
        for (int j = i + 1; j < recv_total; ++j) {
            long long dcx = recvbuf[i].cell_x - recvbuf[j].cell_x;
            long long dcy = recvbuf[i].cell_y - recvbuf[j].cell_y;
            if (dcx < -1 || dcx > 1 || dcy < -1 || dcy > 1) {
                continue;
            }
            double dx = recvbuf[i].x - recvbuf[j].x;
            double dy = recvbuf[i].y - recvbuf[j].y;
            if (dx * dx + dy * dy <= a2) {
                rc = add_edge(&edges, &n_edges, &edges_cap, recvbuf[i].micro_cluster_id, recvbuf[j].micro_cluster_id);
                if (rc != 0) {
                    free(recvbuf);
                    free(edges);
                    alphaagg_result_free(&micro);
                    return rc;
                }
            }
        }
    }
    free(recvbuf);

    int local_edge_vals = n_edges * 2;
    int *all_edge_counts = NULL;
    int *all_edge_displs = NULL;
    long long *recv_edge_vals = NULL;

    if (rank == 0) {
        all_edge_counts = (int *)calloc((size_t)size, sizeof(int));
        all_edge_displs = (int *)calloc((size_t)size, sizeof(int));
        if (!all_edge_counts || !all_edge_displs) {
            free(all_edge_counts);
            free(all_edge_displs);
            free(edges);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    MPI_Gather(&local_edge_vals, 1, MPI_INT, all_edge_counts, 1, MPI_INT, 0, comm);
    int total_edge_vals = 0;
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            all_edge_displs[r] = all_edge_displs[r - 1] + all_edge_counts[r - 1];
        }
        total_edge_vals = all_edge_displs[size - 1] + all_edge_counts[size - 1];
        if (total_edge_vals > 0) {
            recv_edge_vals = (long long *)malloc((size_t)total_edge_vals * sizeof(long long));
            if (!recv_edge_vals) {
                free(all_edge_counts);
                free(all_edge_displs);
                free(edges);
                alphaagg_result_free(&micro);
                return 2;
            }
        }
    }

    MPI_Gatherv((long long *)edges,
                local_edge_vals,
                MPI_LONG_LONG,
                recv_edge_vals,
                all_edge_counts,
                all_edge_displs,
                MPI_LONG_LONG,
                0,
                comm);
    free(edges);

    int *micro_to_global = NULL;
    int global_cluster_count = 0;

    if (rank == 0) {
        alphaagg_dsu_node_t *nodes = NULL;
        rc = alphaagg_dsu_init(&nodes, total_micro);
        if (rc != 0) {
            free(all_edge_counts);
            free(all_edge_displs);
            free(recv_edge_vals);
            return rc;
        }
        for (int i = 0; i + 1 < total_edge_vals; i += 2) {
            long long a = recv_edge_vals[i];
            long long b = recv_edge_vals[i + 1];
            if (a >= 0 && b >= 0 && a < total_micro && b < total_micro) {
                alphaagg_dsu_union(nodes, a, b);
            }
        }
        free(recv_edge_vals);
        free(all_edge_counts);
        free(all_edge_displs);

        micro_to_global = (int *)malloc((size_t)total_micro * sizeof(int));
        int *root_to_label = (int *)malloc((size_t)total_micro * sizeof(int));
        if ((!micro_to_global && total_micro > 0) || (!root_to_label && total_micro > 0)) {
            free(micro_to_global);
            free(root_to_label);
            free(nodes);
            return 2;
        }
        for (long long i = 0; i < total_micro; ++i) {
            root_to_label[i] = -1;
        }
        for (long long i = 0; i < total_micro; ++i) {
            long long r = alphaagg_dsu_find(nodes, i);
            if (root_to_label[r] < 0) {
                root_to_label[r] = global_cluster_count++;
            }
            micro_to_global[i] = root_to_label[r];
        }
        free(root_to_label);
        free(nodes);
    }

    MPI_Bcast(&global_cluster_count, 1, MPI_INT, 0, comm);
    if (rank != 0 && total_micro > 0) {
        micro_to_global = (int *)malloc((size_t)total_micro * sizeof(int));
        if (!micro_to_global) {
            alphaagg_result_free(&micro);
            return 2;
        }
    }
    if (total_micro > 0) {
        MPI_Bcast(micro_to_global, (int)total_micro, MPI_INT, 0, comm);
    }

    int *labels = NULL;
    if (local_n > 0) {
        labels = (int *)malloc((size_t)local_n * sizeof(int));
        if (!labels) {
            free(micro_to_global);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    long long *local_count = NULL;
    double *local_sum_x = NULL;
    double *local_sum_y = NULL;
    double *local_sum_sq = NULL;
    if (global_cluster_count > 0) {
        local_count = (long long *)calloc((size_t)global_cluster_count, sizeof(long long));
        local_sum_x = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        local_sum_y = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        local_sum_sq = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        if (!local_count || !local_sum_x || !local_sum_y || !local_sum_sq) {
            free(micro_to_global);
            free(labels);
            free(local_count);
            free(local_sum_x);
            free(local_sum_y);
            free(local_sum_sq);
            alphaagg_result_free(&micro);
            return 2;
        }
    }

    for (int i = 0; i < local_n; ++i) {
        long long mgid = micro_offset + (long long)micro.labels[i];
        int gid = micro_to_global[mgid];
        labels[i] = gid;
        double x = local_points[2 * i];
        double y = local_points[2 * i + 1];
        local_count[gid] += 1;
        local_sum_x[gid] += x;
        local_sum_y[gid] += y;
        local_sum_sq[gid] += x * x + y * y;
    }

    long long *global_count = NULL;
    double *global_sum_x = NULL;
    double *global_sum_y = NULL;
    double *global_sum_sq = NULL;
    alphaagg_cluster_t *clusters = NULL;
    if (global_cluster_count > 0) {
        global_count = (long long *)calloc((size_t)global_cluster_count, sizeof(long long));
        global_sum_x = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        global_sum_y = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        global_sum_sq = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        clusters = (alphaagg_cluster_t *)calloc((size_t)global_cluster_count, sizeof(alphaagg_cluster_t));
        if (!global_count || !global_sum_x || !global_sum_y || !global_sum_sq || !clusters) {
            free(micro_to_global);
            free(labels);
            free(local_count);
            free(local_sum_x);
            free(local_sum_y);
            free(local_sum_sq);
            free(global_count);
            free(global_sum_x);
            free(global_sum_y);
            free(global_sum_sq);
            free(clusters);
            alphaagg_result_free(&micro);
            return 2;
        }

        MPI_Allreduce(local_count, global_count, global_cluster_count, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(local_sum_x, global_sum_x, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_sum_y, global_sum_y, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_sum_sq, global_sum_sq, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
    }

    free(local_count);
    free(local_sum_x);
    free(local_sum_y);
    free(local_sum_sq);
    free(micro_to_global);
    alphaagg_result_free(&micro);

    double total_sse = 0.0;
    for (int c = 0; c < global_cluster_count; ++c) {
        clusters[c].label = c;
        clusters[c].count = (int)global_count[c];
        if (global_count[c] > 0) {
            double inv = 1.0 / (double)global_count[c];
            clusters[c].centroid_x = global_sum_x[c] * inv;
            clusters[c].centroid_y = global_sum_y[c] * inv;
            clusters[c].seed_x = clusters[c].centroid_x;
            clusters[c].seed_y = clusters[c].centroid_y;
            clusters[c].sse = global_sum_sq[c] - (double)global_count[c] *
                                                  (clusters[c].centroid_x * clusters[c].centroid_x +
                                                   clusters[c].centroid_y * clusters[c].centroid_y);
            if (clusters[c].sse < 0.0 && clusters[c].sse > -1e-12) {
                clusters[c].sse = 0.0;
            }
            total_sse += clusters[c].sse;
        }
    }

    free(global_count);
    free(global_sum_x);
    free(global_sum_y);
    free(global_sum_sq);

    local_out->n_points = local_n;
    local_out->n_clusters = global_cluster_count;
    local_out->labels = labels;
    local_out->clusters = clusters;
    local_out->total_sse = total_sse;
    return 0;
}

int alphaagg_mpi_ptga_aggregate_2d(const double *local_points,
                                   int local_n,
                                   double alpha,
                                   alphaagg_sort_t sorting,
                                   MPI_Comm comm,
                                   alphaagg_result_t *local_out) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!local_out || local_n < 0 || alpha <= 0.0 || sorting < ALPHAAGG_SORT_2_NORM || sorting > ALPHAAGG_SORT_LEXI) {
        return 1;
    }
    if (local_n > 0 && !local_points) {
        return 1;
    }
    alphaagg_result_clear(local_out);

    alphaagg_result_t local_micro = {0};
    int rc = alphaagg_serial_aggregate_2d(local_points, local_n, alpha, sorting, &local_micro);
    if (rc != 0) {
        return rc;
    }

    int local_g = local_micro.n_clusters;
    alphaagg_seed_record_t *local_seeds = NULL;
    if (local_g > 0) {
        local_seeds = (alphaagg_seed_record_t *)malloc((size_t)local_g * sizeof(alphaagg_seed_record_t));
        if (!local_seeds) {
            alphaagg_result_free(&local_micro);
            return 2;
        }
        for (int c = 0; c < local_g; ++c) {
            local_seeds[c].x = local_micro.clusters[c].seed_x;
            local_seeds[c].y = local_micro.clusters[c].seed_y;
            local_seeds[c].local_cluster_id = c;
        }
    }

    int *seed_counts = NULL;
    int *seed_displs = NULL;
    if (rank == 0) {
        seed_counts = (int *)calloc((size_t)size, sizeof(int));
        seed_displs = (int *)calloc((size_t)size, sizeof(int));
        if (!seed_counts || !seed_displs) {
            free(seed_counts);
            free(seed_displs);
            free(local_seeds);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }

    MPI_Gather(&local_g, 1, MPI_INT, seed_counts, 1, MPI_INT, 0, comm);

    int total_g = 0;
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            seed_displs[r] = seed_displs[r - 1] + seed_counts[r - 1];
        }
        total_g = seed_displs[size - 1] + seed_counts[size - 1];
    }
    MPI_Bcast(&total_g, 1, MPI_INT, 0, comm);

    alphaagg_seed_record_t *all_seeds = NULL;
    if (rank == 0 && total_g > 0) {
        all_seeds = (alphaagg_seed_record_t *)malloc((size_t)total_g * sizeof(alphaagg_seed_record_t));
        if (!all_seeds) {
            free(seed_counts);
            free(seed_displs);
            free(local_seeds);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }

    MPI_Datatype seed_type;
    MPI_Type_contiguous((int)sizeof(alphaagg_seed_record_t), MPI_BYTE, &seed_type);
    MPI_Type_commit(&seed_type);
    MPI_Gatherv(local_seeds, local_g, seed_type, all_seeds, seed_counts, seed_displs, seed_type, 0, comm);

    int global_cluster_count = 0;
    int *all_seed_labels = NULL;
    double *global_seed_x = NULL;
    double *global_seed_y = NULL;

    if (rank == 0) {
        if (total_g > 0) {
            double *seed_points = (double *)malloc((size_t)total_g * 2u * sizeof(double));
            all_seed_labels = (int *)malloc((size_t)total_g * sizeof(int));
            if (!seed_points || !all_seed_labels) {
                free(seed_points);
                free(all_seed_labels);
                free(all_seeds);
                free(seed_counts);
                free(seed_displs);
                free(local_seeds);
                MPI_Type_free(&seed_type);
                alphaagg_result_free(&local_micro);
                return 2;
            }
            for (int i = 0; i < total_g; ++i) {
                seed_points[2 * i] = all_seeds[i].x;
                seed_points[2 * i + 1] = all_seeds[i].y;
            }

            alphaagg_result_t seed_global = {0};
            rc = alphaagg_serial_aggregate_2d(seed_points, total_g, alpha, sorting, &seed_global);
            free(seed_points);
            if (rc != 0) {
                free(all_seed_labels);
                free(all_seeds);
                free(seed_counts);
                free(seed_displs);
                free(local_seeds);
                MPI_Type_free(&seed_type);
                alphaagg_result_free(&local_micro);
                return rc;
            }

            global_cluster_count = seed_global.n_clusters;
            global_seed_x = (double *)malloc((size_t)global_cluster_count * sizeof(double));
            global_seed_y = (double *)malloc((size_t)global_cluster_count * sizeof(double));
            if ((!global_seed_x || !global_seed_y) && global_cluster_count > 0) {
                free(all_seed_labels);
                free(all_seeds);
                free(seed_counts);
                free(seed_displs);
                free(local_seeds);
                free(global_seed_x);
                free(global_seed_y);
                MPI_Type_free(&seed_type);
                alphaagg_result_free(&seed_global);
                alphaagg_result_free(&local_micro);
                return 2;
            }

            for (int i = 0; i < total_g; ++i) {
                all_seed_labels[i] = seed_global.labels[i];
            }
            for (int c = 0; c < global_cluster_count; ++c) {
                global_seed_x[c] = seed_global.clusters[c].seed_x;
                global_seed_y[c] = seed_global.clusters[c].seed_y;
            }
            alphaagg_result_free(&seed_global);
        }
        free(all_seeds);
    }

    MPI_Bcast(&global_cluster_count, 1, MPI_INT, 0, comm);

    int *local_cluster_to_global = NULL;
    if (local_g > 0) {
        local_cluster_to_global = (int *)malloc((size_t)local_g * sizeof(int));
        if (!local_cluster_to_global) {
            free(seed_counts);
            free(seed_displs);
            free(local_seeds);
            free(all_seed_labels);
            free(global_seed_x);
            free(global_seed_y);
            MPI_Type_free(&seed_type);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }

    MPI_Scatterv(all_seed_labels,
                 seed_counts,
                 seed_displs,
                 MPI_INT,
                 local_cluster_to_global,
                 local_g,
                 MPI_INT,
                 0,
                 comm);
    MPI_Type_free(&seed_type);

    if (rank != 0 && global_cluster_count > 0) {
        global_seed_x = (double *)malloc((size_t)global_cluster_count * sizeof(double));
        global_seed_y = (double *)malloc((size_t)global_cluster_count * sizeof(double));
        if (!global_seed_x || !global_seed_y) {
            free(seed_counts);
            free(seed_displs);
            free(local_seeds);
            free(all_seed_labels);
            free(global_seed_x);
            free(global_seed_y);
            free(local_cluster_to_global);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }
    if (global_cluster_count > 0) {
        MPI_Bcast(global_seed_x, global_cluster_count, MPI_DOUBLE, 0, comm);
        MPI_Bcast(global_seed_y, global_cluster_count, MPI_DOUBLE, 0, comm);
    }

    free(seed_counts);
    free(seed_displs);
    free(local_seeds);
    free(all_seed_labels);

    int *labels = NULL;
    if (local_n > 0) {
        labels = (int *)malloc((size_t)local_n * sizeof(int));
        if (!labels) {
            free(global_seed_x);
            free(global_seed_y);
            free(local_cluster_to_global);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }

    long long *local_count = NULL;
    double *local_sum_x = NULL;
    double *local_sum_y = NULL;
    double *local_sum_sq = NULL;
    if (global_cluster_count > 0) {
        local_count = (long long *)calloc((size_t)global_cluster_count, sizeof(long long));
        local_sum_x = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        local_sum_y = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        local_sum_sq = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        if (!local_count || !local_sum_x || !local_sum_y || !local_sum_sq) {
            free(global_seed_x);
            free(global_seed_y);
            free(local_cluster_to_global);
            free(labels);
            free(local_count);
            free(local_sum_x);
            free(local_sum_y);
            free(local_sum_sq);
            alphaagg_result_free(&local_micro);
            return 2;
        }
    }

    for (int i = 0; i < local_n; ++i) {
        int gid = local_cluster_to_global[local_micro.labels[i]];
        labels[i] = gid;
        double x = local_points[2 * i];
        double y = local_points[2 * i + 1];
        local_count[gid] += 1;
        local_sum_x[gid] += x;
        local_sum_y[gid] += y;
        local_sum_sq[gid] += x * x + y * y;
    }

    free(local_cluster_to_global);
    alphaagg_result_free(&local_micro);

    long long *global_count = NULL;
    double *global_sum_x = NULL;
    double *global_sum_y = NULL;
    double *global_sum_sq = NULL;
    alphaagg_cluster_t *clusters = NULL;
    if (global_cluster_count > 0) {
        global_count = (long long *)calloc((size_t)global_cluster_count, sizeof(long long));
        global_sum_x = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        global_sum_y = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        global_sum_sq = (double *)calloc((size_t)global_cluster_count, sizeof(double));
        clusters = (alphaagg_cluster_t *)calloc((size_t)global_cluster_count, sizeof(alphaagg_cluster_t));
        if (!global_count || !global_sum_x || !global_sum_y || !global_sum_sq || !clusters) {
            free(global_seed_x);
            free(global_seed_y);
            free(labels);
            free(local_count);
            free(local_sum_x);
            free(local_sum_y);
            free(local_sum_sq);
            free(global_count);
            free(global_sum_x);
            free(global_sum_y);
            free(global_sum_sq);
            free(clusters);
            return 2;
        }

        MPI_Allreduce(local_count, global_count, global_cluster_count, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(local_sum_x, global_sum_x, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_sum_y, global_sum_y, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(local_sum_sq, global_sum_sq, global_cluster_count, MPI_DOUBLE, MPI_SUM, comm);
    }

    free(local_count);
    free(local_sum_x);
    free(local_sum_y);
    free(local_sum_sq);

    double total_sse = 0.0;
    for (int c = 0; c < global_cluster_count; ++c) {
        clusters[c].label = c;
        clusters[c].count = (int)global_count[c];
        clusters[c].seed_x = global_seed_x[c];
        clusters[c].seed_y = global_seed_y[c];
        if (global_count[c] > 0) {
            double inv = 1.0 / (double)global_count[c];
            clusters[c].centroid_x = global_sum_x[c] * inv;
            clusters[c].centroid_y = global_sum_y[c] * inv;
            clusters[c].sse = global_sum_sq[c] - (double)global_count[c] *
                                                  (clusters[c].centroid_x * clusters[c].centroid_x +
                                                   clusters[c].centroid_y * clusters[c].centroid_y);
            if (clusters[c].sse < 0.0 && clusters[c].sse > -1e-12) {
                clusters[c].sse = 0.0;
            }
            total_sse += clusters[c].sse;
        }
    }

    free(global_seed_x);
    free(global_seed_y);
    free(global_count);
    free(global_sum_x);
    free(global_sum_y);
    free(global_sum_sq);

    local_out->n_points = local_n;
    local_out->n_clusters = global_cluster_count;
    local_out->labels = labels;
    local_out->clusters = clusters;
    local_out->total_sse = total_sse;
    return 0;
}
