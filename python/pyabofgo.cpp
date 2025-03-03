
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "../include/FactorGraph.h"
#include "../include/GNSSFactor.h"
#include "../include/IMUFactor.h"
#include "../include/optimizer.h"

namespace py = pybind11;

PYBIND11_MODULE(pyabofgo, m) {
    m.doc() = "Python wrapper for AboFGO GNSS/IMU fusion library";

    py::class_<AboFGO::FactorGraph>(m, "FactorGraph")
            .def(py::init<>())
            .def("add_gnss_factor", [](AboFGO::FactorGraph &graph, const Eigen::Vector3d &measurement, double weight) {
                graph.addFactor(new AboFGO::GNSSFactor(measurement, weight));
            }, py::arg("measurement"), py::arg("weight"))
            .def("add_imu_factor", [](AboFGO::FactorGraph &graph, const Eigen::Vector3d &measurement, double weight) {
                graph.addFactor(new AboFGO::IMUFactor(measurement, weight));
            }, py::arg("measurement"), py::arg("weight"))
            .def("get_state", [](const AboFGO::FactorGraph &graph) {
                return graph.getState();
            })
            .def("compute_total_error", &AboFGO::FactorGraph::computeTotalError);

    py::class_<AboFGO::Optimizer>(m, "Optimizer")
            .def(py::init<AboFGO::FactorGraph *>())
            .def("optimize", &AboFGO::Optimizer::optimize, py::arg("iterations") = 10);
}
