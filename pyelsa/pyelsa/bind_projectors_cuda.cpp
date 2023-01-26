#include <pybind11/pybind11.h>

#include "JosephsMethodCUDA.h"
#include "SiddonsMethodCUDA.h"
#include "VoxelProjectorCUDA.h"

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

    py::class_<elsa::BlobVoxelProjectorCUDA<float>, elsa::LinearOperator<float>>
        BlobVoxelProjectorCUDAf(m, "BlobVoxelProjectorCUDAf");
    BlobVoxelProjectorCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("BlobVoxelProjectorCUDA") = m.attr("BlobVoxelProjectorCUDAf");

    py::class_<elsa::BlobVoxelProjectorCUDA<double>, elsa::LinearOperator<double>>
        BlobVoxelProjectorCUDAd(m, "BlobVoxelProjectorCUDAd");
    BlobVoxelProjectorCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::BSplineVoxelProjectorCUDA<float>, elsa::LinearOperator<float>>
        BSplineVoxelProjectorCUDAf(m, "BSplineVoxelProjectorCUDAf");
    BSplineVoxelProjectorCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("BSplineVoxelProjectorCUDA") = m.attr("BSplineVoxelProjectorCUDAf");

    py::class_<elsa::BSplineVoxelProjectorCUDA<double>, elsa::LinearOperator<double>>
        BSplineVoxelProjectorCUDAd(m, "BSplineVoxelProjectorCUDAd");
    BSplineVoxelProjectorCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));
    py::class_<elsa::PhaseContrastBlobVoxelProjectorCUDA<float>, elsa::LinearOperator<float>>
        PhaseContrastBlobVoxelProjectorCUDAf(m, "PhaseContrastBlobVoxelProjectorCUDAf");
    PhaseContrastBlobVoxelProjectorCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("PhaseContrastBlobVoxelProjectorCUDA") = m.attr("PhaseContrastBlobVoxelProjectorCUDAf");

    py::class_<elsa::PhaseContrastBlobVoxelProjectorCUDA<double>, elsa::LinearOperator<double>>
        PhaseContrastBlobVoxelProjectorCUDAd(m, "PhaseContrastBlobVoxelProjectorCUDAd");
    PhaseContrastBlobVoxelProjectorCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::PhaseContrastBSplineVoxelProjectorCUDA<float>, elsa::LinearOperator<float>>
        PhaseContrastBSplineVoxelProjectorCUDAf(m, "PhaseContrastBSplineVoxelProjectorCUDAf");
    PhaseContrastBSplineVoxelProjectorCUDAf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("PhaseContrastBSplineVoxelProjectorCUDA") =
        m.attr("PhaseContrastBSplineVoxelProjectorCUDAf");

    py::class_<elsa::PhaseContrastBSplineVoxelProjectorCUDA<double>, elsa::LinearOperator<double>>
        PhaseContrastBSplineVoxelProjectorCUDAd(m, "PhaseContrastBSplineVoxelProjectorCUDAd");
    PhaseContrastBSplineVoxelProjectorCUDAd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    elsa::ProjectorsCUDAHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_projectors_cuda, m)
{
    add_definitions_pyelsa_projectors_cuda(m);
}
