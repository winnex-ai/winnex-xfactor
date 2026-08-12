/**
 * py_module.cpp — Python bindings for winnex-xfactor (pybind11).
 *
 * Exposes the deterministic manifold operator:
 *   compute_xfactor(embed_tokens, vocab, dim, tau) → (P, rank, variance)
 *   XFactor.project / project_batch
 *   expand_spectral
 *
 *   import winnex_xfactor
 *   P, rank, var = winnex_xfactor.compute_xfactor(E, vocab, dim, tau=0.95)
 */
#include "winnex_xfactor/x_factor.hpp"

#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using winnex_xfactor::XFactor;

PYBIND11_MODULE(_winnex_xfactor, m) {
    m.doc() = "winnex-xfactor — deterministic manifold embedding (O(D^2 r) power iteration)";

    py::class_<XFactor>(m, "XFactor")
        .def(py::init<const float*, int, int, double>(),
             py::arg("embed_tokens"), py::arg("vocab"), py::arg("dim"),
             py::arg("variance_tau") = 0.95)
        .def_property_readonly("dim", &XFactor::dim)
        .def_property_readonly("rank", &XFactor::rank)
        .def_property_readonly("variance_captured", &XFactor::variance_captured)
        .def("project", [](const XFactor& self, py::array_t<float, py::array::c_style | py::array::forcecast> v) {
            auto info = v.request();
            if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                throw std::runtime_error("XFactor.project: input must be length dim");
            py::array_t<float> out(self.dim());
            self.project((const float*)info.ptr, out.mutable_data());
            return out;
        }, py::arg("v"))
        .def("project_batch", [](const XFactor& self, py::array_t<float, py::array::c_style | py::array::forcecast> v) {
            auto info = v.request();
            if (info.ndim != 2 || (int)info.shape[1] != self.dim())
                throw std::runtime_error("XFactor.project_batch: input must be (n, dim)");
            int n = (int)info.shape[0];
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)n, (py::ssize_t)self.dim()});
            self.project_batch((const float*)info.ptr, out.mutable_data(), n);
            return out;
        }, py::arg("v"))
        .def("projector", [](const XFactor& self) {
            int D = self.dim();
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)D, (py::ssize_t)D});
            std::memcpy(out.mutable_data(), self.projector(), (size_t)D * D * sizeof(float));
            return out;
        })
        .def("basis", [](const XFactor& self) {
            int D = self.dim(), r = self.rank();
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)D, (py::ssize_t)r});
            std::memcpy(out.mutable_data(), self.basis(), (size_t)D * (size_t)r * sizeof(float));
            return out;
        });

    // Free function: build the X-factor projector from a sample of the model's
    // embed_tokens. Returns (projector P [D×D], rank, variance_captured).
    m.def("compute_xfactor", [](py::array_t<float, py::array::c_style | py::array::forcecast> E,
                                int vocab, int dim, double tau) {
        auto info = E.request();
        if (info.ndim != 2) throw std::runtime_error("compute_xfactor: embed_tokens must be 2D");
        XFactor xf((const float*)info.ptr, vocab, dim, tau);
        const int D = xf.dim();
        py::array_t<float> P(std::vector<py::ssize_t>{(py::ssize_t)D, (py::ssize_t)D});
        std::memcpy(P.mutable_data(), xf.projector(), (size_t)D * D * sizeof(float));
        return py::make_tuple(P, xf.rank(), xf.variance_captured());
    }, py::arg("embed_tokens"), py::arg("vocab"), py::arg("dim"),
       py::arg("variance_tau") = 0.95);

    m.def("expand_spectral", [](py::array_t<float, py::array::c_style | py::array::forcecast> psi,
                                int d, int D) {
        auto info = psi.request();
        if (info.ndim != 1 || (int)info.shape[0] != d)
            throw std::runtime_error("expand_spectral: input must be length d");
        py::array_t<float> out(D);
        winnex_xfactor::expand_spectral((const float*)info.ptr, d, out.mutable_data(), D);
        return out;
    }, py::arg("psi"), py::arg("d"), py::arg("D"));
}
