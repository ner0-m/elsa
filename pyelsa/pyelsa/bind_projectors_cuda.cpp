#include <pybind11/pybind11.h>

#include "JosephsMethodCUDA.h"
#include "SiddonsMethodCUDA.h"

#include "hints/projectors_cuda_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_projectors_cuda(py::module& m)
{
    py::class_<elsa::SiddonsMethodCUDA<float>, elsa::LinearOperator<float>> SiddonsMethodCUDAf(
        m, "SiddonsMethodCUDAf");
    SiddonsMethodCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("SiddonsMethodCUDA") = m.attr("SiddonsMethodCUDAf");

    py::class_<elsa::SiddonsMethodCUDA<double>, elsa::LinearOperator<double>> SiddonsMethodCUDAd(
        m, "SiddonsMethodCUDAd");
    SiddonsMethodCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::JosephsMethodCUDA<float>, elsa::LinearOperator<float>> JosephsMethodCUDAf(
        m, "JosephsMethodCUDAf");
    JosephsMethodCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&, bool>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
        py::arg("fast") = static_cast<bool>(true));

    m.attr("JosephsMethodCUDA") = m.attr("JosephsMethodCUDAf");

    py::class_<elsa::JosephsMethodCUDA<double>, elsa::LinearOperator<double>> JosephsMethodCUDAd(
        m, "JosephsMethodCUDAd");
    JosephsMethodCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&, bool>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
        py::arg("fast") = static_cast<bool>(true));

    elsa::ProjectorsCUDAHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_projectors_cuda, m)
{
    add_definitions_pyelsa_projectors_cuda(m);
}
