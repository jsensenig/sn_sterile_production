//
// Created by Jon Sensenig on 4/11/25.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(test_pybind, m) {
        m.def("distribution_interpolation", []() { return 5; });
    }
