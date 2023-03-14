#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "BSplines.h"
#include "BinaryMethod.h"
#include "JosephsMethod.h"
#include "SiddonsMethod.h"
#include "VoxelProjector.h"
#include "SiddonsMethodBranchless.h"
#include "SubsetSampler.h"

#include "hints/projectors_hints.cpp"

namespace py = pybind11;

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

    py::class_<elsa::BinaryMethod<float>, elsa::LinearOperator<float>> BinaryMethodf(
        m, "BinaryMethodf");
    BinaryMethodf.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                      py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("BinaryMethod") = m.attr("BinaryMethodf");

    py::class_<elsa::BinaryMethod<double>, elsa::LinearOperator<double>> BinaryMethodd(
        m, "BinaryMethodd");
    BinaryMethodd.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                      py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::SiddonsMethod<float>, elsa::LinearOperator<float>> SiddonsMethodf(
        m, "SiddonsMethodf");
    SiddonsMethodf.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                       py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("SiddonsMethod") = m.attr("SiddonsMethodf");

    py::class_<elsa::SiddonsMethod<double>, elsa::LinearOperator<double>> SiddonsMethodd(
        m, "SiddonsMethodd");
    SiddonsMethodd.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                       py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::BlobVoxelProjector<float>, elsa::LinearOperator<float>> BlobVoxelProjectorf(
        m, "BlobVoxelProjectorf");
    BlobVoxelProjectorf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("BlobVoxelProjector") = m.attr("BlobVoxelProjectorf");

    py::class_<elsa::BlobVoxelProjector<double>, elsa::LinearOperator<double>> BlobVoxelProjectord(
        m, "BlobVoxelProjectord");
    BlobVoxelProjectord.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::BSplineVoxelProjector<float>, elsa::LinearOperator<float>>
        BSplineVoxelProjectorf(m, "BSplineVoxelProjectorf");
    BSplineVoxelProjectorf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("BSplineVoxelProjector") = m.attr("BSplineVoxelProjectorf");

    py::class_<elsa::BSplineVoxelProjector<double>, elsa::LinearOperator<double>>
        BSplineVoxelProjectord(m, "BSplineVoxelProjectord");
    BSplineVoxelProjectord.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));
    py::class_<elsa::PhaseContrastBlobVoxelProjector<float>, elsa::LinearOperator<float>>
        PhaseContrastBlobVoxelProjectorf(m, "PhaseContrastBlobVoxelProjectorf");
    PhaseContrastBlobVoxelProjectorf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("PhaseContrastBlobVoxelProjector") = m.attr("PhaseContrastBlobVoxelProjectorf");

    py::class_<elsa::PhaseContrastBlobVoxelProjector<double>, elsa::LinearOperator<double>>
        PhaseContrastBlobVoxelProjectord(m, "PhaseContrastBlobVoxelProjectord");
    PhaseContrastBlobVoxelProjectord.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::PhaseContrastBSplineVoxelProjector<float>, elsa::LinearOperator<float>>
        PhaseContrastBSplineVoxelProjectorf(m, "PhaseContrastBSplineVoxelProjectorf");
    PhaseContrastBSplineVoxelProjectorf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("PhaseContrastBSplineVoxelProjector") = m.attr("PhaseContrastBSplineVoxelProjectorf");

    py::class_<elsa::PhaseContrastBSplineVoxelProjector<double>, elsa::LinearOperator<double>>
        PhaseContrastBSplineVoxelProjectord(m, "PhaseContrastBSplineVoxelProjectord");
    PhaseContrastBSplineVoxelProjectord.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::SiddonsMethodBranchless<float>, elsa::LinearOperator<float>>
        SiddonsMethodBranchlessf(m, "SiddonsMethodBranchlessf");
    SiddonsMethodBranchlessf.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("SiddonsMethodBranchless") = m.attr("SiddonsMethodBranchlessf");

    py::class_<elsa::SiddonsMethodBranchless<double>, elsa::LinearOperator<double>>
        SiddonsMethodBranchlessd(m, "SiddonsMethodBranchlessd");
    SiddonsMethodBranchlessd.def(
        py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
        py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    py::class_<elsa::JosephsMethod<float>, elsa::LinearOperator<float>> JosephsMethodf(
        m, "JosephsMethodf");
    JosephsMethodf.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                       py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

    m.attr("JosephsMethod") = m.attr("JosephsMethodf");

    py::class_<elsa::JosephsMethod<double>, elsa::LinearOperator<double>> JosephsMethodd(
        m, "JosephsMethodd");
    JosephsMethodd.def(py::init<const elsa::VolumeDescriptor&, const elsa::DetectorDescriptor&>(),
                       py::arg("domainDescriptor"), py::arg("rangeDescriptor"));

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
    //              elsa::SubsetSampler<elsa::PlanarDetectorDescriptor, thrust::complex<float>>::*)(
    //              const elsa::DataContainer<thrust::complex<float>>&))(
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
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long,
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
    //     .def(py::init<const elsa::VolumeDescriptor&, const elsa::PlanarDetectorDescriptor&, long,
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
