#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../core/combinatorics.hpp"
#include "../core/fock.hpp"
#include "../core/photonics_core.cpp"

namespace py = pybind11;
using namespace pqc;   // C++ types live in pqc:: so they cannot clash with other pybind11 modules

namespace {

std::vector<cplx_t> square_matrix(py::array_t<cplx_t, py::array::c_style | py::array::forcecast> m, int& n) {
    if (m.ndim() != 2 || m.shape(0) != m.shape(1)) throw std::invalid_argument("matrix must be square");
    n = static_cast<int>(m.shape(0));
    return std::vector<cplx_t>(m.data(), m.data() + n * n);
}

py::tuple to_tuple(const Occupation& occ) {
    py::tuple t(occ.size());
    for (size_t k = 0; k < occ.size(); ++k) t[k] = occ[k];
    return t;
}

}  // namespace

PYBIND11_MODULE(photonics_core, m) {
    py::class_<Photon>(m, "Photon")
        .def(py::init<std::string, std::string, int, double>(),
             py::arg("mode"), py::arg("polarization"),
             py::arg("pair_id") = -1, py::arg("timestamp") = 0.0)
        .def("info", &Photon::info)
        .def_readwrite("mode", &Photon::mode)
        .def_readwrite("polarization", &Photon::polarization)
        .def_readwrite("pair_id", &Photon::pair_id)
        .def_readwrite("timestamp", &Photon::timestamp);

    py::class_<Component, std::shared_ptr<Component>>(m, "Component");

    py::class_<Beamsplitter, Component, std::shared_ptr<Beamsplitter>>(m, "Beamsplitter")
        .def(py::init<>());

    py::class_<Detector, Component, std::shared_ptr<Detector>>(m, "Detector")
        .def(py::init<>())
        .def("count", &Detector::count)
        .def("reset", &Detector::reset);

    py::class_<SPDCSource, Component, std::shared_ptr<SPDCSource>>(m, "SPDCSource")
        .def(py::init<>());

    py::class_<Circuit>(m, "Circuit")
        .def(py::init<>())
        .def("add_component", &Circuit::add_component)
        .def("run", &Circuit::run, py::arg("photons") = std::vector<Photon>{});

    m.def("permanent", [](py::array_t<cplx_t, py::array::c_style | py::array::forcecast> a) {
        int n; auto v = square_matrix(a, n); return permanent_ryser(v, n);
    }, py::arg("matrix"), "Permanent (Ryser with Gray code)");
    m.def("hafnian", [](py::array_t<cplx_t, py::array::c_style | py::array::forcecast> a) {
        int n; auto v = square_matrix(a, n); return hafnian_dp(v, n);
    }, py::arg("matrix"), "Hafnian of a symmetric matrix (subset dynamic programming)");

    py::class_<FockState>(m, "FockState")
        .def(py::init<const Occupation&>(), py::arg("occupation"),
             "Basis state |n_0, n_1, ...> with the given photon number in each mode")
        .def_property_readonly("n_modes", &FockState::n_modes)
        .def("amplitude", &FockState::amplitude, py::arg("occupation"))
        .def("probability", &FockState::probability, py::arg("occupation"))
        .def("norm", &FockState::norm)
        .def("mean_photon_number", &FockState::mean_photon_number, py::arg("mode"))
        .def("amplitudes", [](const FockState& s) {
            py::dict d;
            for (const auto& [occ, a] : s.terms()) d[to_tuple(occ)] = a;
            return d;
        }, "Dict mapping occupation tuples to complex amplitudes")
        .def("probabilities", [](const FockState& s) {
            py::dict d;
            for (const auto& [occ, a] : s.terms()) d[to_tuple(occ)] = std::norm(a);
            return d;
        }, "Dict mapping occupation tuples to detection probabilities")
        .def("beamsplitter", &FockState::apply_beamsplitter,
             py::arg("i"), py::arg("j"), py::arg("theta") = M_PI / 4, py::arg("phi") = 0.0)
        .def("phase_shift", &FockState::apply_phase_shift, py::arg("mode"), py::arg("phi"))
        .def("sample", [](const FockState& s, int shots, unsigned seed) {
            py::list out;
            for (const auto& occ : s.sample(shots, seed)) out.append(to_tuple(occ));
            return out;
        }, py::arg("shots"), py::arg("seed") = 0)
        .def("__repr__", [](const FockState& s) {
            std::string r;
            for (const auto& [occ, a] : s.terms()) {
                if (!r.empty()) r += " + ";
                r += "(" + std::to_string(a.real()) + (a.imag() < 0 ? "-" : "+") +
                     std::to_string(std::abs(a.imag())) + "j)|";
                for (size_t k = 0; k < occ.size(); ++k) r += (k ? "," : "") + std::to_string(occ[k]);
                r += ">";
            }
            return "FockState(" + r + ")";
        });

    py::class_<OpticalCircuit>(m, "OpticalCircuit")
        .def(py::init<int>(), py::arg("n_modes"))
        .def_property_readonly("n_modes", &OpticalCircuit::n_modes)
        .def("__len__", &OpticalCircuit::size)
        .def("beamsplitter", &OpticalCircuit::beamsplitter,
             py::arg("i"), py::arg("j"), py::arg("theta") = M_PI / 4, py::arg("phi") = 0.0,
             py::return_value_policy::reference_internal)
        .def("phase_shift", &OpticalCircuit::phase_shift, py::arg("mode"), py::arg("phi"),
             py::return_value_policy::reference_internal)
        .def("run", &OpticalCircuit::run, py::arg("state"));
}
