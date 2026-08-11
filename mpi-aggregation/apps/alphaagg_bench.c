#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mpi_alphaagg/mpi_alphaagg.h"

static unsigned int xorshift32(unsigned int *state) {
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double urand(unsigned int *state) {
    return (double)xorshift32(state) / (double)UINT32_MAX;
}

static double gauss(unsigned int *state) {
    double u1 = urand(state);
    double u2 = urand(state);
    if (u1 < 1e-12) {
        u1 = 1e-12;
    }
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static int parse_sorting(const char *s, alphaagg_sort_t *out) {
    if (strcmp(s, "2-norm") == 0) {
        *out = ALPHAAGG_SORT_2_NORM;
    } else if (strcmp(s, "1-norm") == 0) {
        *out = ALPHAAGG_SORT_1_NORM;
    } else if (strcmp(s, "lexi") == 0) {
        *out = ALPHAAGG_SORT_LEXI;
    } else {
        return 1;
    }
    return 0;
}

static int generate_points(double *pts,
                           int local_n,
                           long long global_start,
                           long long global_n,
                           const char *dataset,
                           unsigned int seed,
                           int rank) {
    unsigned int st = seed + (unsigned int)(rank * 9973);
    if (strcmp(dataset, "uniform") == 0) {
        for (int i = 0; i < local_n; ++i) {
            pts[2 * i] = urand(&st);
            pts[2 * i + 1] = urand(&st);
        }
    } else if (strcmp(dataset, "blobs") == 0) {
        const double c[4][2] = {{0.2, 0.2}, {0.2, 0.8}, {0.8, 0.2}, {0.8, 0.8}};
        for (int i = 0; i < local_n; ++i) {
            int k = (int)(urand(&st) * 4.0) % 4;
            pts[2 * i] = c[k][0] + 0.05 * gauss(&st);
            pts[2 * i + 1] = c[k][1] + 0.05 * gauss(&st);
        }
    } else if (strcmp(dataset, "grid") == 0) {
        long long side = (long long)sqrt((double)(global_n > 1 ? global_n : 1));
        if (side < 1) {
            side = 1;
        }
        for (int i = 0; i < local_n; ++i) {
            long long gid = global_start + i;
            long long gx = gid % side;
            long long gy = gid / side;
            pts[2 * i] = (double)gx / (double)(side > 1 ? side - 1 : 1) + 0.01 * gauss(&st);
            pts[2 * i + 1] = (double)gy / (double)(side > 1 ? side - 1 : 1) + 0.01 * gauss(&st);
        }
    } else {
        return 1;
    }
    return 0;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        return 0;
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 1000000;
    double alpha = 0.05;
    const char *sorting_str = "2-norm";
    const char *dataset = "blobs";
    const char *algorithm = "ptga";
    unsigned int seed = 42;
    int repeat = 5;
    const char *csv = "benchmark.csv";

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sorting") == 0 && i + 1 < argc) {
            sorting_str = argv[++i];
        } else if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            dataset = argv[++i];
        } else if (strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csv = argv[++i];
        }
    }

    alphaagg_sort_t sorting;
    if (parse_sorting(sorting_str, &sorting) != 0 || n < 0 || alpha <= 0.0 || repeat < 1 ||
        (strcmp(algorithm, "ptga") != 0 && strcmp(algorithm, "grid") != 0)) {
        if (rank == 0) {
            fprintf(stderr, "invalid args\n");
        }
        MPI_Finalize();
        return 1;
    }

    long long base = n / size;
    long long rem = n % size;
    int local_n = (int)(base + (rank < rem ? 1 : 0));
    long long local_start = rank * base + (rank < rem ? rank : rem);

    double *points = (double *)malloc((size_t)local_n * 2u * sizeof(double));
    if (!points && local_n > 0) {
        MPI_Finalize();
        return 2;
    }

    if (generate_points(points, local_n, local_start, n, dataset, seed, rank) != 0) {
        if (rank == 0) {
            fprintf(stderr, "failed generating dataset\n");
        }
        free(points);
        MPI_Finalize();
        return 1;
    }

    FILE *fp = NULL;
    if (rank == 0) {
        int exists = file_exists(csv);
        fp = fopen(csv, exists ? "a" : "w");
        if (!fp) {
            free(points);
            MPI_Finalize();
            return 2;
        }
        if (!exists) {
            fprintf(fp,
                    "frontend,algorithm,n_points,alpha,sorting,dataset,seed,comm_size,rank_count,repeat_index,wall_time_sec,n_local,n_global_clusters,total_sse\n");
        }
    }

    for (int r = 0; r < repeat; ++r) {
        alphaagg_result_t out = {0};
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int rc = 0;
        if (strcmp(algorithm, "grid") == 0) {
            rc = alphaagg_mpi_aggregate_2d(points, local_n, alpha, sorting, MPI_COMM_WORLD, &out);
        } else {
            rc = alphaagg_mpi_ptga_aggregate_2d(points, local_n, alpha, sorting, MPI_COMM_WORLD, &out);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        if (rc != 0) {
            if (rank == 0) {
                fprintf(stderr, "aggregation failed: %d\n", rc);
            }
            alphaagg_result_free(&out);
            if (fp) {
                fclose(fp);
            }
            free(points);
            MPI_Finalize();
            return rc;
        }

        if (rank == 0 && fp) {
            fprintf(fp,
                    "c,mpi_alphaagg_%s,%lld,%.10g,%s,%s,%u,%d,%d,%d,%.9f,%d,%d,%.9f\n",
                    algorithm,
                    n,
                    alpha,
                    sorting_str,
                    dataset,
                    seed,
                    size,
                    size,
                    r,
                    t1 - t0,
                    local_n,
                    out.n_clusters,
                    out.total_sse);
            fflush(fp);
        }
        alphaagg_result_free(&out);
    }

    if (fp) {
        fclose(fp);
    }
    free(points);
    MPI_Finalize();
    return 0;
}
