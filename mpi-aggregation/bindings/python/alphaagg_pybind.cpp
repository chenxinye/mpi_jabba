#include <mpi.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

extern "C" {
#include "mpi_alphaagg/alphaagg.h"
#include "mpi_alphaagg/mpi_alphaagg.h"
}

namespace py = pybind11;

static alphaagg_sort_t parse_sorting(const std::string &sorting) {
    if (sorting == "2-norm") return ALPHAAGG_SORT_2_NORM;
    if (sorting == "1-norm") return ALPHAAGG_SORT_1_NORM;
    if (sorting == "lexi") return ALPHAAGG_SORT_LEXI;
    throw std::invalid_argument("sorting must be one of: 2-norm, 1-norm, lexi");
}

static py::dict result_to_dict(alphaagg_result_t &res) {
    py::array_t<int> labels(res.n_points);
    auto l = labels.mutable_unchecked<1>();
    for (int i = 0; i < res.n_points; ++i) l(i) = res.labels ? res.labels[i] : -1;

    py::array_t<double> clusters({res.n_clusters, 7});
    auto c = clusters.mutable_unchecked<2>();
    for (int i = 0; i < res.n_clusters; ++i) {
        c(i, 0) = (double)res.clusters[i].label;
        c(i, 1) = (double)res.clusters[i].count;
        c(i, 2) = res.clusters[i].seed_x;
        c(i, 3) = res.clusters[i].seed_y;
        c(i, 4) = res.clusters[i].centroid_x;
        c(i, 5) = res.clusters[i].centroid_y;
        c(i, 6) = res.clusters[i].sse;
    }

    py::dict out;
    out["labels"] = labels;
    out["n_clusters"] = py::int_(res.n_clusters);
    out["total_sse"] = py::float_(res.total_sse);
    out["clusters"] = clusters;
    return out;
}

static int g_mpi_initialized_here = 0;

static void finalize_mpi_at_exit() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && g_mpi_initialized_here) {
        MPI_Finalize();
    }
}

static void ensure_mpi_initialized() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
        g_mpi_initialized_here = 1;
        std::atexit(finalize_mpi_at_exit);
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "MPI alpha aggregation core bindings";

    m.def("aggregate_serial", [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
                                  double alpha,
                                  const std::string &sorting) {
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }
        int n = (int)buf.shape[0];
        alphaagg_result_t res = {0};
        int rc = alphaagg_serial_aggregate_2d((const double *)buf.ptr, n, alpha, parse_sorting(sorting), &res);
        if (rc != 0) {
            throw std::runtime_error("alphaagg_serial_aggregate_2d failed: " + std::to_string(rc));
        }
        py::dict out = result_to_dict(res);
        alphaagg_result_free(&res);
        return out;
    }, py::arg("points"), py::arg("alpha"), py::arg("sorting") = "2-norm");

    m.def("aggregate_mpi", [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
                               double alpha,
                               const std::string &sorting) {
        ensure_mpi_initialized();
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }
        int n = (int)buf.shape[0];
        alphaagg_result_t res = {0};
        int rc = alphaagg_mpi_aggregate_2d((const double *)buf.ptr,
                                           n,
                                           alpha,
                                           parse_sorting(sorting),
                                           MPI_COMM_WORLD,
                                           &res);
        if (rc != 0) {
            throw std::runtime_error("alphaagg_mpi_aggregate_2d failed: " + std::to_string(rc));
        }
        py::dict out = result_to_dict(res);
        alphaagg_result_free(&res);
        return out;
    }, py::arg("points"), py::arg("alpha"), py::arg("sorting") = "2-norm");

    m.def("aggregate_mpi_grid", [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
                                    double alpha,
                                    const std::string &sorting) {
        ensure_mpi_initialized();
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }
        int n = (int)buf.shape[0];
        alphaagg_result_t res = {0};
        int rc = alphaagg_mpi_aggregate_2d((const double *)buf.ptr,
                                           n,
                                           alpha,
                                           parse_sorting(sorting),
                                           MPI_COMM_WORLD,
                                           &res);
        if (rc != 0) {
            throw std::runtime_error("alphaagg_mpi_aggregate_2d failed: " + std::to_string(rc));
        }
        py::dict out = result_to_dict(res);
        alphaagg_result_free(&res);
        return out;
    }, py::arg("points"), py::arg("alpha"), py::arg("sorting") = "2-norm");

    m.def("aggregate_mpi_ptga", [](py::array_t<double, py::array::c_style | py::array::forcecast> points,
                                    double alpha,
                                    const std::string &sorting) {
        ensure_mpi_initialized();
        auto buf = points.request();
        if (buf.ndim != 2 || buf.shape[1] != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }
        int n = (int)buf.shape[0];
        alphaagg_result_t res = {0};
        int rc = alphaagg_mpi_ptga_aggregate_2d((const double *)buf.ptr,
                                                n,
                                                alpha,
                                                parse_sorting(sorting),
                                                MPI_COMM_WORLD,
                                                &res);
        if (rc != 0) {
            throw std::runtime_error("alphaagg_mpi_ptga_aggregate_2d failed: " + std::to_string(rc));
        }
        py::dict out = result_to_dict(res);
        alphaagg_result_free(&res);
        return out;
    }, py::arg("points"), py::arg("alpha"), py::arg("sorting") = "2-norm");
}
