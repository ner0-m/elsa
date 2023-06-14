#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "BinaryMethod.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"
#include "VoxelProjector.h"
#include "PhaseContrastProjector.h"
#include "LutProjector.h"
#include "SubsetSampler.h"
#include "Blobs.h"
#include "BSplines.h"

#include "bind_common.h"

#include "hints/projectors_hints.cpp"

namespace py = pybind11;
using index_t = std::ptrdiff_t; ///< global type for indices

template <typename ProjectorClassFloat, typename ProjectorClassDouble>
void bind_blob_projector(py::module& m, const char* name)
{
    std::string float_name = std::string(name) + "f";
    std::string double_name = std::string(name) + "d";

    py::class_<ProjectorClassFloat, elsa::LinearOperator<float>> projector_float(
        m, float_name.c_str());
    projector_float.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&,
                                 float, float, index_t>(),
                        py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
                        py::arg("radius") = elsa::blobs::DEFAULT_RADIUS,
                        py::arg("alpha") = elsa::blobs::DEFAULT_ALPHA,
                        py::arg("order") = elsa::blobs::DEFAULT_ORDER);

    m.attr(name) = m.attr(float_name.c_str());

    py::class_<ProjectorClassDouble, elsa::LinearOperator<double>> projector_double(
        m, double_name.c_str());
    projector_double.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&,
                                  double, double, index_t>(),
                         py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
                         py::arg("radius") = elsa::blobs::DEFAULT_RADIUS,
                         py::arg("alpha") = elsa::blobs::DEFAULT_ALPHA,
                         py::arg("order") = elsa::blobs::DEFAULT_ORDER);
}

template <typename ProjectorClassFloat, typename ProjectorClassDouble>
void bind_bspline_projector(py::module& m, const char* name)
{
    std::string float_name = std::string(name) + "f";
    std::string double_name = std::string(name) + "d";

    py::class_<ProjectorClassFloat, elsa::LinearOperator<float>> projector_float(
        m, float_name.c_str());
    projector_float.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&, index_t>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
        py::arg("order") = elsa::bspline::DEFAULT_ORDER);

    m.attr(name) = m.attr(float_name.c_str());

    py::class_<ProjectorClassDouble, elsa::LinearOperator<double>> projector_double(
        m, double_name.c_str());
    projector_double.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&, index_t>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"),
        py::arg("order") = elsa::bspline::DEFAULT_ORDER);
}

template <typename ProjectorClassFloat, typename ProjectorClassDouble>
void bind_projector(py::module& m, const char* name)
{
    std::string float_name = std::string(name) + "f";
    std::string double_name = std::string(name) + "d";

    py::class_<ProjectorClassFloat, elsa::LinearOperator<float>> projector_float(
        m, float_name.c_str());
    projector_float.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr(name) = m.attr(float_name.c_str());

    py::class_<ProjectorClassDouble, elsa::LinearOperator<double>> projector_double(
        m, double_name.c_str());
    projector_double.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                         py::arg("domainDescriptor"), py::arg("rangeDescriptor"));
}

void add_definitions_pyelsa_projectors(py::module& m)
{
    py::enum_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>::SamplingStrategy>(
        m, "SubsetSamplerPlanarDetectorDescriptorfloatSamplingStrategy")
        .value("ROTATIONAL_CLUSTERING",
               elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                   float>::SamplingStrategy::ROTATIONAL_CLUSTERING)
        .value("ROUND_ROBIN", elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                                  float>::SamplingStrategy::ROUND_ROBIN);

    py::enum_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>::SamplingStrategy>(
        m, "SubsetSamplerPlanarDetectorDescriptordoubleSamplingStrategy")
        .value("ROTATIONAL_CLUSTERING",
               elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                   double>::SamplingStrategy::ROTATIONAL_CLUSTERING)
        .value("ROUND_ROBIN", elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                                  double>::SamplingStrategy::ROUND_ROBIN);

    py::enum_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                  thrust::complex<float>>::SamplingStrategy>(
        m, "SubsetSamplerPlanarDetectorDescriptorcfSamplingStrategy")
        .value("ROTATIONAL_CLUSTERING",
               elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                   thrust::complex<float>>::SamplingStrategy::ROTATIONAL_CLUSTERING)
        .value("ROUND_ROBIN",
               elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                   thrust::complex<float>>::SamplingStrategy::ROUND_ROBIN);

    py::enum_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                  thrust::complex<double>>::SamplingStrategy>(
        m, "SubsetSamplerPlanarDetectorDescriptorcdSamplingStrategy")
        .value(
            "ROTATIONAL_CLUSTERING",
            elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                thrust::complex<double>>::SamplingStrategy::ROTATIONAL_CLUSTERING)
        .value("ROUND_ROBIN",
               elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                   thrust::complex<double>>::SamplingStrategy::ROUND_ROBIN);

    bind_projector<elsa::BinaryMethod<float>, elsa::BinaryMethod<double>>(m, "BinaryMethod");
    bind_projector<elsa::SiddonsMethod<float>, elsa::SiddonsMethod<double>>(m, "SiddonsMethod");
    bind_projector<elsa::JosephsMethod<float>, elsa::JosephsMethod<double>>(m, "JosephsMethod");

    bind_blob_projector<elsa::BlobProjector<float>, elsa::BlobProjector<double>>(m,
                                                                                 "BlobProjector");
    bind_blob_projector<elsa::BlobVoxelProjector<float>, elsa::BlobVoxelProjector<double>>(
        m, "BlobVoxelProjector");
    bind_blob_projector<elsa::PhaseContrastBlobVoxelProjector<float>,
                        elsa::PhaseContrastBlobVoxelProjector<double>>(
        m, "PhaseContrastBlobVoxelProjector");

    bind_bspline_projector<elsa::BSplineProjector<float>, elsa::BSplineProjector<double>>(
        m, "BSplineProjector");
    bind_bspline_projector<elsa::BSplineVoxelProjector<float>, elsa::BSplineVoxelProjector<double>>(
        m, "BSplineVoxelProjector");
    bind_bspline_projector<elsa::PhaseContrastBSplineVoxelProjector<float>,
                           elsa::PhaseContrastBSplineVoxelProjector<double>>(
        m, "PhaseContrastBSplineVoxelProjector");

    py::class_<elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>>
        CloneableSubsetSamplerPlanarDetectorDescriptorfloat(
            m, "CloneableSubsetSamplerPlanarDetectorDescriptorfloat");
    CloneableSubsetSamplerPlanarDetectorDescriptorfloat
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::*)(
                 const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>&)
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::*)(
                 const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>&)
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>,
                 std::default_delete<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>>(
                 elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::*)()
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>::clone));

    py::class_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>,
               elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>>>
        SubsetSamplerPlanarDetectorDescriptorfloat(m, "SubsetSamplerPlanarDetectorDescriptorfloat");
    SubsetSamplerPlanarDetectorDescriptorfloat
        .def("getPartitionedData",
             (elsa::DataContainer<float>(
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>::*)(
                 const elsa::DataContainer<float>&))(
                 &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>::getPartitionedData),
             py::arg("sinogram"), py::return_value_policy::move)
        .def_static("splitRotationalClustering",
                    (std::vector<std::vector<long, std::allocator<long>>,
                                 std::allocator<std::vector<long, std::allocator<long>>>>(*)(
                        const elsa::PlanarDetectorDescriptor&, long))(
                        &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                             float>::splitRotationalClustering),
                    py::arg("detectorDescriptor"), py::arg("nSubsets"),
                    py::return_value_policy::move)
        .def_static(
            "splitRoundRobin",
            (std::vector<std::vector<long, std::allocator<long>>,
                         std::allocator<std::vector<long, std::allocator<long>>>>(*)(
                const std::vector<long, std::allocator<long>>&, long))(
                &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>::splitRoundRobin),
            py::arg("indices"), py::arg("nSubsets"), py::return_value_policy::move)
        .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long>(),
             py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"))
        .def(py::init<
                 const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long,
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, float>::SamplingStrategy>(),
             py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"),
             py::arg("samplingStrategy"));

    elsa::SubsetSamplerHints<elsa::PlanarDetectorDescriptor, float>::addCustomMethods(
        SubsetSamplerPlanarDetectorDescriptorfloat);

    py::class_<elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>>
        CloneableSubsetSamplerPlanarDetectorDescriptordouble(
            m, "CloneableSubsetSamplerPlanarDetectorDescriptordouble");
    CloneableSubsetSamplerPlanarDetectorDescriptordouble
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::*)(
                 const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>&)
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::*)(
                 const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>&)
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>,
                 std::default_delete<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>>(
                 elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::*)()
                  const)(&elsa::Cloneable<
                         elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>::clone));

    py::class_<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>,
               elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>>>
        SubsetSamplerPlanarDetectorDescriptordouble(m,
                                                    "SubsetSamplerPlanarDetectorDescriptordouble");
    SubsetSamplerPlanarDetectorDescriptordouble
        .def("getPartitionedData",
             (elsa::DataContainer<double>(
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>::*)(
                 const elsa::DataContainer<double>&))(
                 &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>::getPartitionedData),
             py::arg("sinogram"), py::return_value_policy::move)
        .def_static("splitRotationalClustering",
                    (std::vector<std::vector<long, std::allocator<long>>,
                                 std::allocator<std::vector<long, std::allocator<long>>>>(*)(
                        const elsa::PlanarDetectorDescriptor&, long))(
                        &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
                                             double>::splitRotationalClustering),
                    py::arg("detectorDescriptor"), py::arg("nSubsets"),
                    py::return_value_policy::move)
        .def_static(
            "splitRoundRobin",
            (std::vector<std::vector<long, std::allocator<long>>,
                         std::allocator<std::vector<long, std::allocator<long>>>>(*)(
                const std::vector<long, std::allocator<long>>&, long))(
                &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>::splitRoundRobin),
            py::arg("indices"), py::arg("nSubsets"), py::return_value_policy::move)
        .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long>(),
             py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"))
        .def(py::init<
                 const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long,
                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, double>::SamplingStrategy>(),
             py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"),
             py::arg("samplingStrategy"));

    elsa::SubsetSamplerHints<elsa::PlanarDetectorDescriptor, double>::addCustomMethods(
        SubsetSamplerPlanarDetectorDescriptordouble);

    // py::class_<
    //     elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //     thrust::complex<float>>>> CloneableSubsetSamplerPlanarDetectorDescriptorcf(
    //         m, "CloneableSubsetSamplerPlanarDetectorDescriptorcf");
    // CloneableSubsetSamplerPlanarDetectorDescriptorcf
    //     .def("__ne__",
    //          (bool(elsa::Cloneable<
    //                elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                thrust::complex<float>>>::*)(
    //              const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<float>>&)
    //               const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                           thrust::complex<float>>>::operator!=),
    //          py::arg("other"))
    //     .def("__eq__",
    //          (bool(elsa::Cloneable<
    //                elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                thrust::complex<float>>>::*)(
    //              const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<float>>&)
    //               const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                           thrust::complex<float>>>::operator==),
    //          py::arg("other"))
    //     .def("clone",
    //          (std::unique_ptr<
    //              elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, thrust::complex<float>>,
    //              std::default_delete<
    //                  elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                  thrust::complex<float>>>>(
    //              elsa::Cloneable<
    //                  elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                  thrust::complex<float>>>::*)()
    //               const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                           thrust::complex<float>>>::clone));
    //
    // py::class_<
    //     elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, thrust::complex<float>>,
    //     elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //     thrust::complex<float>>>> SubsetSamplerPlanarDetectorDescriptorcf(m,
    //     "SubsetSamplerPlanarDetectorDescriptorcf");
    // SubsetSamplerPlanarDetectorDescriptorcf
    //     .def("getPartitionedData",
    //          (elsa::DataContainer<thrust::complex<float>>(
    //              elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<float>>::*)( const
    //              elsa::DataContainer<thrust::complex<float>>&))(
    //              &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                   thrust::complex<float>>::getPartitionedData),
    //          py::arg("sinogram"), py::return_value_policy::move)
    //     .def_static("splitRotationalClustering",
    //                 (std::vector<std::vector<long, std::allocator<long>>,
    //                              std::allocator<std::vector<long, std::allocator<long>>>>(*)(
    //                     const elsa::PlanarDetectorDescriptor&, long))(
    //                     &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                          thrust::complex<float>>::splitRotationalClustering),
    //                 py::arg("detectorDescriptor"), py::arg("nSubsets"),
    //                 py::return_value_policy::move)
    //     .def_static("splitRoundRobin",
    //                 (std::vector<std::vector<long, std::allocator<long>>,
    //                              std::allocator<std::vector<long, std::allocator<long>>>>(*)(
    //                     const std::vector<long, std::allocator<long>>&, long))(
    //                     &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                          thrust::complex<float>>::splitRoundRobin),
    //                 py::arg("indices"), py::arg("nSubsets"), py::return_value_policy::move)
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&,
    //     long>(),
    //          py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"))
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&,
    //     long,
    //                   elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                       thrust::complex<float>>::SamplingStrategy>(),
    //          py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"),
    //          py::arg("samplingStrategy"));
    //
    // py::class_<
    //     elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //     thrust::complex<double>>>> CloneableSubsetSamplerPlanarDetectorDescriptorcd(
    //         m, "CloneableSubsetSamplerPlanarDetectorDescriptorcd");
    // CloneableSubsetSamplerPlanarDetectorDescriptorcd
    //     .def("__ne__",
    //          (bool(elsa::Cloneable<
    //                elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                thrust::complex<double>>>::*)(
    //              const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<double>>&)
    //               const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                           thrust::complex<double>>>::operator!=),
    //          py::arg("other"))
    //     .def("__eq__",
    //          (bool(elsa::Cloneable<
    //                elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                thrust::complex<double>>>::*)(
    //              const elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<double>>&)
    //               const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                           thrust::complex<double>>>::operator==),
    //          py::arg("other"))
    //     .def(
    //         "clone",
    //         (std::unique_ptr<
    //             elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, thrust::complex<double>>,
    //             std::default_delete<
    //                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                 thrust::complex<double>>>>(
    //             elsa::Cloneable<
    //                 elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                 thrust::complex<double>>>::*)()
    //              const)(&elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                                          thrust::complex<double>>>::clone));
    //
    // py::class_<
    //     elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, thrust::complex<double>>,
    //     elsa::Cloneable<elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //     thrust::complex<double>>>> SubsetSamplerPlanarDetectorDescriptorcd(m,
    //     "SubsetSamplerPlanarDetectorDescriptorcd");
    // SubsetSamplerPlanarDetectorDescriptorcd
    //     .def("getPartitionedData",
    //          (elsa::DataContainer<thrust::complex<double>>(
    //              elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //              thrust::complex<double>>::*)( const
    //              elsa::DataContainer<thrust::complex<double>>&))(
    //              &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                   thrust::complex<double>>::getPartitionedData),
    //          py::arg("sinogram"), py::return_value_policy::move)
    //     .def_static("splitRotationalClustering",
    //                 (std::vector<std::vector<long, std::allocator<long>>,
    //                              std::allocator<std::vector<long, std::allocator<long>>>>(*)(
    //                     const elsa::PlanarDetectorDescriptor&, long))(
    //                     &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                          thrust::complex<double>>::splitRotationalClustering),
    //                 py::arg("detectorDescriptor"), py::arg("nSubsets"),
    //                 py::return_value_policy::move)
    //     .def_static("splitRoundRobin",
    //                 (std::vector<std::vector<long, std::allocator<long>>,
    //                              std::allocator<std::vector<long, std::allocator<long>>>>(*)(
    //                     const std::vector<long, std::allocator<long>>&, long))(
    //                     &elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                          thrust::complex<double>>::splitRoundRobin),
    //                 py::arg("indices"), py::arg("nSubsets"), py::return_value_policy::move)
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&,
    //     long>(),
    //          py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"))
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&,
    //     long,
    //                   elsa::SubsetSampler<elsa::PlanarDetectorDescriptor,
    //                                       thrust::complex<double>>::SamplingStrategy>(),
    //          py::arg("volumeDescriptor"), py::arg("detectorDescriptor"), py::arg("nSubsets"),
    //          py::arg("samplingStrategy"));

    elsa::ProjectorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_projectors, m)
{
    add_definitions_pyelsa_projectors(m);
}
