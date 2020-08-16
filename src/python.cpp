#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "distance.h"
#include "parameters.h"
#include "index_random.h"
#include "index_ssg.h"

#define NUMPY_C_STYLE py::array::c_style | py:array::forcecast

namespace py = pybind11;

using efanna2e::Metric;
using efanna2e::Parameters;
using efanna2e::Index;
using efanna2e::IndexRandom;
using efanna2e::IndexSSG;

using array = py::array_t<float, py::array::c_style | py::array::forcecast>;

void set_seed(int seed) {
    srand(seed);
}

PYBIND11_MODULE(pyssg, m) {
    m.def("set_seed", &set_seed, "Set C++ random seed");
    m.def("load_data", &efanna2e::load_data, "Load data from SIFT-style binary file");

    // Metric
    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("INNER_PRODUCT", Metric::INNER_PRODUCT)
        .value("FAST_L2", Metric::FAST_L2)
        .value("PQ", Metric::PQ);

    // Parameters
    // py::class_<Parameters>(m, "Parameters")
    //     .def("__getitem__", [](const Parameters& params, std::string key) {
    //         try { return params.GetRaw(key); }
    //         catch (const std::invalid_argument&) {
    //             throw py::key_error("Key '" + key + "' does not exist");
    //         }
    //     })
    //     .def("__setitem__", [](const Parameters& params,
    //                            std::string key, unsigned value) {
    //         params.Set<unsigned>(key, value);
    //     })
    //     .def("__setitem__", [](const Parameters& params,
    //                            std::string key, float value) {
    //         params.Set<float>(key, value);
    //     });

    // IndexRandom
    // py::class_<IndexRandom>(m, "IndexRandom")
    //     .def(py::init<size_t, size_t>())
    //     .def("build", &IndexRandom::Build)
    //     .def("search", [](const IndexRandom& index,
    //                       const float* query, const float* x,
    //                       size_t k, unsigned *indices) {
    //         Parameters params;
    //         index.Search(query, x, k, params, indices);
    //     });

    // IndexSSG
    py::class_<IndexSSG>(m, "IndexSSG")
        .def(py::init([](size_t dim, size_t num_data,
                         Metric metric=Metric::FAST_L2) {
            IndexRandom init_index(dim, num_data);
            IndexSSG* index = new IndexSSG(dim, num_data, metric, &init_index);
            return index;
        }), py::arg("dim"), py::arg("num_data"), py::arg("metric") = Metric::FAST_L2)

        // .def("build", &IndexSSG::Build)  # Currently only search

        /* Load SSG graph along with data */
        .def("load", [](IndexSSG& index,
                        std::string graph, array data) {
            index.Load(graph.c_str());
            index.OptimizeGraph(data.data());
        })

        /* Save graph to file */
        .def("save", &IndexSSG::Save)

        /* Do KNN search
            @param query: an 1-D numpy array represents query
            @param k: number of neighbors to search for
            @param l: L parameter for search algorithm

            @return a list contains K neighbors' indices (start from 0)
         */
        .def("search", [](IndexSSG& index, array query, size_t k, unsigned l)
                          -> std::vector<unsigned> {
            // NOTE: This lambda function will convert numpy array to raw pointer

            if (query.ndim() != 1) {
                throw py::value_error("Query should be 1-D array");
            }
            if (query.shape()[0] != index.GetDimension()) {
                throw py::value_error("Dimension mismatch");
            }

            // Construct Parameters object
            Parameters params;
            params.Set<unsigned>("L_search", l);

            // Result vector, will be converted to list by pybind11 automatically
            std::vector<unsigned> indices(k);

            // Do Search
            index.SearchWithOptGraph(query.data(), k, params, indices.data());

            return indices;
        });
}