#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "Cloneable.h"
#include "DataContainer.h"
#include "Descriptors/BlockDescriptor.h"
#include "Descriptors/DataDescriptor.h"
#include "Descriptors/DetectorDescriptor.h"
#include "Descriptors/IdenticalBlocksDescriptor.h"
#include "Descriptors/PartitionDescriptor.h"
#include "Descriptors/CurvedDetectorDescriptor.h"
#include "Descriptors/PlanarDetectorDescriptor.h"
#include "Descriptors/RandomBlocksDescriptor.h"
#include "Descriptors/VolumeDescriptor.h"
#include "Geometry.h"
#include "LinearOperator.h"
#include "StrongTypes.h"
#include "Utilities/FormatConfig.h"

#include "hints/core_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_core(py::module& m)
{
    py::enum_<elsa::DataHandlerType>(m, "DataHandlerType")
        .value("CPU", elsa::DataHandlerType::CPU)
        .value("GPU", elsa::DataHandlerType::GPU)
        .value("MAP_CPU", elsa::DataHandlerType::MAP_CPU)
        .value("MAP_GPU", elsa::DataHandlerType::MAP_GPU);

    py::enum_<elsa::FFTNorm>(m, "FFTNorm")
        .value("BACKWARD", elsa::FFTNorm::BACKWARD)
        .value("FORWARD", elsa::FFTNorm::FORWARD)
        .value("ORTHO", elsa::FFTNorm::ORTHO);

    py::class_<elsa::Cloneable<elsa::DataDescriptor>> CloneableDataDescriptor(
        m, "CloneableDataDescriptor");
    CloneableDataDescriptor
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::DataDescriptor>::*)(const elsa::DataDescriptor&)
                  const)(&elsa::Cloneable<elsa::DataDescriptor>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::DataDescriptor>::*)(const elsa::DataDescriptor&)
                  const)(&elsa::Cloneable<elsa::DataDescriptor>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::DataDescriptor, std::default_delete<elsa::DataDescriptor>>(
                 elsa::Cloneable<elsa::DataDescriptor>::*)()
                  const)(&elsa::Cloneable<elsa::DataDescriptor>::clone));

    py::class_<elsa::DataDescriptor, elsa::Cloneable<elsa::DataDescriptor>> DataDescriptor(
        m, "DataDescriptor");
    DataDescriptor
        .def("getLocationOfOrigin",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::DataDescriptor::*)()
                  const)(&elsa::DataDescriptor::getLocationOfOrigin),
             py::return_value_policy::move)
        .def("getSpacingPerDimension",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::DataDescriptor::*)()
                  const)(&elsa::DataDescriptor::getSpacingPerDimension),
             py::return_value_policy::move)
        .def("getCoordinateFromIndex",
             (Eigen::Matrix<long, -1, 1, 0, -1, 1>(elsa::DataDescriptor::*)(long)
                  const)(&elsa::DataDescriptor::getCoordinateFromIndex),
             py::arg("index"), py::return_value_policy::move)
        .def("getNumberOfCoefficientsPerDimension",
             (Eigen::Matrix<long, -1, 1, 0, -1, 1>(elsa::DataDescriptor::*)()
                  const)(&elsa::DataDescriptor::getNumberOfCoefficientsPerDimension),
             py::return_value_policy::move)
        .def("getIndexFromCoordinate",
             (long(elsa::DataDescriptor::*)(const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataDescriptor::getIndexFromCoordinate),
             py::arg("coordinate"))
        .def("getNumberOfCoefficients", (long(elsa::DataDescriptor::*)()
                                             const)(&elsa::DataDescriptor::getNumberOfCoefficients))
        .def("getNumberOfDimensions",
             (long(elsa::DataDescriptor::*)() const)(&elsa::DataDescriptor::getNumberOfDimensions));

    py::class_<elsa::format_config> format_config(m, "format_config");
    format_config
        .def("set",
             (elsa::format_config & (elsa::format_config::*) (const elsa::format_config&) )(
                 &elsa::format_config::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::format_config&>());

    py::class_<elsa::DataContainer<float>> DataContainerf(m, "DataContainerf",
                                                          py::buffer_protocol());
    DataContainerf
        .def("__ne__",
             (bool(elsa::DataContainer<float>::*)(const elsa::DataContainer<float>&)
                  const)(&elsa::DataContainer<float>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::DataContainer<float>::*)(const elsa::DataContainer<float>&)
                  const)(&elsa::DataContainer<float>::operator==),
             py::arg("other"))
        .def("__imul__",
             (elsa::DataContainer<
                  float> & (elsa::DataContainer<float>::*) (const elsa::DataContainer<float>&) )(
                 &elsa::DataContainer<float>::operator*=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<
                  float> & (elsa::DataContainer<float>::*) (const elsa::DataContainer<float>&) )(
                 &elsa::DataContainer<float>::operator+=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<
                  float> & (elsa::DataContainer<float>::*) (const elsa::DataContainer<float>&) )(
                 &elsa::DataContainer<float>::operator-=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<
                  float> & (elsa::DataContainer<float>::*) (const elsa::DataContainer<float>&) )(
                 &elsa::DataContainer<float>::operator/=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<
                  float> & (elsa::DataContainer<float>::*) (const elsa::DataContainer<float>&) )(
                 &elsa::DataContainer<float>::operator=),
             py::arg("other"), py::return_value_policy::reference_internal)
        .def("__imul__",
             (elsa::DataContainer<float> & (elsa::DataContainer<float>::*) (float) )(
                 &elsa::DataContainer<float>::operator*=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<float> & (elsa::DataContainer<float>::*) (float) )(
                 &elsa::DataContainer<float>::operator+=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<float> & (elsa::DataContainer<float>::*) (float) )(
                 &elsa::DataContainer<float>::operator-=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<float> & (elsa::DataContainer<float>::*) (float) )(
                 &elsa::DataContainer<float>::operator/=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<float> & (elsa::DataContainer<float>::*) (float) )(
                 &elsa::DataContainer<float>::operator=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("viewAs",
             (elsa::DataContainer<float>(elsa::DataContainer<float>::*)(
                 const elsa::DataDescriptor&))(&elsa::DataContainer<float>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (elsa::DataContainer<float>(elsa::DataContainer<float>::*)(long))(
                 &elsa::DataContainer<float>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (elsa::DataContainer<float>(elsa::DataContainer<float>::*)(long))(
                 &elsa::DataContainer<float>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("loadToCPU",
             (elsa::DataContainer<float>(elsa::DataContainer<float>::*)())(
                 &elsa::DataContainer<float>::loadToCPU),
             py::return_value_policy::move)
        .def("loadToGPU",
             (elsa::DataContainer<float>(elsa::DataContainer<float>::*)())(
                 &elsa::DataContainer<float>::loadToGPU),
             py::return_value_policy::move)
        .def("viewAs",
             (const elsa::DataContainer<float> (elsa::DataContainer<float>::*)(
                 const elsa::DataDescriptor&) const)(&elsa::DataContainer<float>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (const elsa::DataContainer<float> (elsa::DataContainer<float>::*)(long)
                  const)(&elsa::DataContainer<float>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (const elsa::DataContainer<float> (elsa::DataContainer<float>::*)(long)
                  const)(&elsa::DataContainer<float>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("getDataDescriptor",
             (const elsa::DataDescriptor& (elsa::DataContainer<float>::*) ()
                  const)(&elsa::DataContainer<float>::getDataDescriptor),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (const float& (elsa::DataContainer<float>::*) (long)
                  const)(&elsa::DataContainer<float>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("getDataHandlerType",
             (elsa::DataHandlerType(elsa::DataContainer<float>::*)()
                  const)(&elsa::DataContainer<float>::getDataHandlerType),
             py::return_value_policy::move)
        .def("__getitem__",
             (float& (elsa::DataContainer<float>::*) (long) )(
                 &elsa::DataContainer<float>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("at",
             (float(elsa::DataContainer<float>::*)(const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataContainer<float>::at),
             py::arg("coordinate"))
        .def("dot",
             (float(elsa::DataContainer<float>::*)(const elsa::DataContainer<float>&)
                  const)(&elsa::DataContainer<float>::dot),
             py::arg("other"))
        .def("l1Norm",
             (float(elsa::DataContainer<float>::*)() const)(&elsa::DataContainer<float>::l1Norm))
        .def("l2Norm",
             (float(elsa::DataContainer<float>::*)() const)(&elsa::DataContainer<float>::l2Norm))
        .def("lInfNorm",
             (float(elsa::DataContainer<float>::*)() const)(&elsa::DataContainer<float>::lInfNorm))
        .def("maxElement", (float(elsa::DataContainer<float>::*)()
                                const)(&elsa::DataContainer<float>::maxElement))
        .def("minElement", (float(elsa::DataContainer<float>::*)()
                                const)(&elsa::DataContainer<float>::minElement))
        .def("squaredL2Norm", (float(elsa::DataContainer<float>::*)()
                                   const)(&elsa::DataContainer<float>::squaredL2Norm))
        .def("sum",
             (float(elsa::DataContainer<float>::*)() const)(&elsa::DataContainer<float>::sum))
        .def("getSize",
             (long(elsa::DataContainer<float>::*)() const)(&elsa::DataContainer<float>::getSize))
        .def("l0PseudoNorm", (long(elsa::DataContainer<float>::*)()
                                  const)(&elsa::DataContainer<float>::l0PseudoNorm))
        .def("format",
             (void(elsa::DataContainer<float>::*)(std::basic_ostream<char, std::char_traits<char>>&)
                  const)(&elsa::DataContainer<float>::format),
             py::arg("os"))
        .def("format",
             (void(elsa::DataContainer<float>::*)(std::basic_ostream<char, std::char_traits<char>>&,
                                                  elsa::format_config)
                  const)(&elsa::DataContainer<float>::format),
             py::arg("os"), py::arg("cfg"))
        .def(py::init<const elsa::DataContainer<float>&>(), py::arg("other"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<float, -1, 1, 0, -1, 1>&,
                      elsa::DataHandlerType>(),
             py::arg("dataDescriptor"), py::arg("data"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def(py::init<const elsa::DataDescriptor&, elsa::DataHandlerType>(),
             py::arg("dataDescriptor"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def("fft",
             (void(elsa::DataContainer<float>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<float>::fft),
             py::arg("norm"))
        .def("ifft",
             (void(elsa::DataContainer<float>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<float>::ifft),
             py::arg("norm"));

    elsa::DataContainerHints<float>::addCustomMethods(DataContainerf);

    elsa::DataContainerHints<float>::exposeBufferInfo(DataContainerf);

    m.attr("DataContainer") = m.attr("DataContainerf");

    py::class_<elsa::DataContainer<std::complex<float>>> DataContainercf(m, "DataContainercf");
    DataContainercf
        .def("__ne__",
             (bool(elsa::DataContainer<std::complex<float>>::*)(
                 const elsa::DataContainer<std::complex<float>>&)
                  const)(&elsa::DataContainer<std::complex<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::DataContainer<std::complex<float>>::*)(
                 const elsa::DataContainer<std::complex<float>>&)
                  const)(&elsa::DataContainer<std::complex<float>>::operator==),
             py::arg("other"))
        .def("__imul__",
             (elsa::DataContainer<std::complex<
                  float>> & (elsa::DataContainer<std::complex<float>>::*) (std::complex<float>) )(
                 &elsa::DataContainer<std::complex<float>>::operator*=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<std::complex<
                  float>> & (elsa::DataContainer<std::complex<float>>::*) (std::complex<float>) )(
                 &elsa::DataContainer<std::complex<float>>::operator+=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<std::complex<
                  float>> & (elsa::DataContainer<std::complex<float>>::*) (std::complex<float>) )(
                 &elsa::DataContainer<std::complex<float>>::operator-=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<std::complex<
                  float>> & (elsa::DataContainer<std::complex<float>>::*) (std::complex<float>) )(
                 &elsa::DataContainer<std::complex<float>>::operator/=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<std::complex<
                  float>> & (elsa::DataContainer<std::complex<float>>::*) (std::complex<float>) )(
                 &elsa::DataContainer<std::complex<float>>::operator=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "__imul__",
            (elsa::DataContainer<std::complex<
                 float>> & (elsa::DataContainer<std::complex<float>>::*) (const elsa::DataContainer<std::complex<float>>&) )(
                &elsa::DataContainer<std::complex<float>>::operator*=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__iadd__",
            (elsa::DataContainer<std::complex<
                 float>> & (elsa::DataContainer<std::complex<float>>::*) (const elsa::DataContainer<std::complex<float>>&) )(
                &elsa::DataContainer<std::complex<float>>::operator+=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__isub__",
            (elsa::DataContainer<std::complex<
                 float>> & (elsa::DataContainer<std::complex<float>>::*) (const elsa::DataContainer<std::complex<float>>&) )(
                &elsa::DataContainer<std::complex<float>>::operator-=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__idiv__",
            (elsa::DataContainer<std::complex<
                 float>> & (elsa::DataContainer<std::complex<float>>::*) (const elsa::DataContainer<std::complex<float>>&) )(
                &elsa::DataContainer<std::complex<float>>::operator/=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::DataContainer<std::complex<
                 float>> & (elsa::DataContainer<std::complex<float>>::*) (const elsa::DataContainer<std::complex<float>>&) )(
                &elsa::DataContainer<std::complex<float>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("viewAs",
             (elsa::DataContainer<std::complex<float>>(elsa::DataContainer<std::complex<float>>::*)(
                 const elsa::DataDescriptor&))(&elsa::DataContainer<std::complex<float>>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (elsa::DataContainer<std::complex<float>>(elsa::DataContainer<std::complex<float>>::*)(
                 long))(&elsa::DataContainer<std::complex<float>>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (elsa::DataContainer<std::complex<float>>(elsa::DataContainer<std::complex<float>>::*)(
                 long))(&elsa::DataContainer<std::complex<float>>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("loadToCPU",
             (elsa::DataContainer<std::complex<float>>(
                 elsa::DataContainer<std::complex<float>>::*)())(
                 &elsa::DataContainer<std::complex<float>>::loadToCPU),
             py::return_value_policy::move)
        .def("loadToGPU",
             (elsa::DataContainer<std::complex<float>>(
                 elsa::DataContainer<std::complex<float>>::*)())(
                 &elsa::DataContainer<std::complex<float>>::loadToGPU),
             py::return_value_policy::move)
        .def("__getitem__",
             (std::complex<float> & (elsa::DataContainer<std::complex<float>>::*) (long) )(
                 &elsa::DataContainer<std::complex<float>>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("at",
             (std::complex<float>(elsa::DataContainer<std::complex<float>>::*)(
                 const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataContainer<std::complex<float>>::at),
             py::arg("coordinate"), py::return_value_policy::move)
        .def("dot",
             (std::complex<float>(elsa::DataContainer<std::complex<float>>::*)(
                 const elsa::DataContainer<std::complex<float>>&)
                  const)(&elsa::DataContainer<std::complex<float>>::dot),
             py::arg("other"), py::return_value_policy::move)
        .def("maxElement",
             (std::complex<float>(elsa::DataContainer<std::complex<float>>::*)()
                  const)(&elsa::DataContainer<std::complex<float>>::maxElement),
             py::return_value_policy::move)
        .def("minElement",
             (std::complex<float>(elsa::DataContainer<std::complex<float>>::*)()
                  const)(&elsa::DataContainer<std::complex<float>>::minElement),
             py::return_value_policy::move)
        .def("sum",
             (std::complex<float>(elsa::DataContainer<std::complex<float>>::*)()
                  const)(&elsa::DataContainer<std::complex<float>>::sum),
             py::return_value_policy::move)
        .def("viewAs",
             (const elsa::DataContainer<std::complex<float>> (
                 elsa::DataContainer<std::complex<float>>::*)(const elsa::DataDescriptor&)
                  const)(&elsa::DataContainer<std::complex<float>>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (const elsa::DataContainer<std::complex<float>> (
                 elsa::DataContainer<std::complex<float>>::*)(long)
                  const)(&elsa::DataContainer<std::complex<float>>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (const elsa::DataContainer<std::complex<float>> (
                 elsa::DataContainer<std::complex<float>>::*)(long)
                  const)(&elsa::DataContainer<std::complex<float>>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("getDataDescriptor",
             (const elsa::DataDescriptor& (elsa::DataContainer<std::complex<float>>::*) ()
                  const)(&elsa::DataContainer<std::complex<float>>::getDataDescriptor),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (const std::complex<float>& (elsa::DataContainer<std::complex<float>>::*) (long)
                  const)(&elsa::DataContainer<std::complex<float>>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("getDataHandlerType",
             (elsa::DataHandlerType(elsa::DataContainer<std::complex<float>>::*)()
                  const)(&elsa::DataContainer<std::complex<float>>::getDataHandlerType),
             py::return_value_policy::move)
        .def("l1Norm", (float(elsa::DataContainer<std::complex<float>>::*)()
                            const)(&elsa::DataContainer<std::complex<float>>::l1Norm))
        .def("l2Norm", (float(elsa::DataContainer<std::complex<float>>::*)()
                            const)(&elsa::DataContainer<std::complex<float>>::l2Norm))
        .def("lInfNorm", (float(elsa::DataContainer<std::complex<float>>::*)()
                              const)(&elsa::DataContainer<std::complex<float>>::lInfNorm))
        .def("squaredL2Norm", (float(elsa::DataContainer<std::complex<float>>::*)()
                                   const)(&elsa::DataContainer<std::complex<float>>::squaredL2Norm))
        .def("getSize", (long(elsa::DataContainer<std::complex<float>>::*)()
                             const)(&elsa::DataContainer<std::complex<float>>::getSize))
        .def("l0PseudoNorm", (long(elsa::DataContainer<std::complex<float>>::*)()
                                  const)(&elsa::DataContainer<std::complex<float>>::l0PseudoNorm))
        .def("format",
             (void(elsa::DataContainer<std::complex<float>>::*)(
                 std::basic_ostream<char, std::char_traits<char>>&)
                  const)(&elsa::DataContainer<std::complex<float>>::format),
             py::arg("os"))
        .def("format",
             (void(elsa::DataContainer<std::complex<float>>::*)(
                 std::basic_ostream<char, std::char_traits<char>>&, elsa::format_config)
                  const)(&elsa::DataContainer<std::complex<float>>::format),
             py::arg("os"), py::arg("cfg"))
        .def(py::init<const elsa::DataContainer<std::complex<float>>&>(), py::arg("other"))
        .def(py::init<const elsa::DataDescriptor&,
                      const Eigen::Matrix<std::complex<float>, -1, 1, 0, -1, 1>&,
                      elsa::DataHandlerType>(),
             py::arg("dataDescriptor"), py::arg("data"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def(py::init<const elsa::DataDescriptor&, elsa::DataHandlerType>(),
             py::arg("dataDescriptor"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def("fft",
             (void(elsa::DataContainer<std::complex<float>>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<std::complex<float>>::fft),
             py::arg("norm"))
        .def("ifft",
             (void(elsa::DataContainer<std::complex<float>>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<std::complex<float>>::ifft),
             py::arg("norm"));

    elsa::DataContainerComplexHints<std::complex<float>>::addCustomMethods(DataContainercf);

    py::class_<elsa::DataContainer<double>> DataContainerd(m, "DataContainerd",
                                                           py::buffer_protocol());
    DataContainerd
        .def("__ne__",
             (bool(elsa::DataContainer<double>::*)(const elsa::DataContainer<double>&)
                  const)(&elsa::DataContainer<double>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::DataContainer<double>::*)(const elsa::DataContainer<double>&)
                  const)(&elsa::DataContainer<double>::operator==),
             py::arg("other"))
        .def("__imul__",
             (elsa::DataContainer<
                  double> & (elsa::DataContainer<double>::*) (const elsa::DataContainer<double>&) )(
                 &elsa::DataContainer<double>::operator*=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<
                  double> & (elsa::DataContainer<double>::*) (const elsa::DataContainer<double>&) )(
                 &elsa::DataContainer<double>::operator+=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<
                  double> & (elsa::DataContainer<double>::*) (const elsa::DataContainer<double>&) )(
                 &elsa::DataContainer<double>::operator-=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<
                  double> & (elsa::DataContainer<double>::*) (const elsa::DataContainer<double>&) )(
                 &elsa::DataContainer<double>::operator/=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<
                  double> & (elsa::DataContainer<double>::*) (const elsa::DataContainer<double>&) )(
                 &elsa::DataContainer<double>::operator=),
             py::arg("other"), py::return_value_policy::reference_internal)
        .def("__imul__",
             (elsa::DataContainer<double> & (elsa::DataContainer<double>::*) (double) )(
                 &elsa::DataContainer<double>::operator*=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<double> & (elsa::DataContainer<double>::*) (double) )(
                 &elsa::DataContainer<double>::operator+=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<double> & (elsa::DataContainer<double>::*) (double) )(
                 &elsa::DataContainer<double>::operator-=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<double> & (elsa::DataContainer<double>::*) (double) )(
                 &elsa::DataContainer<double>::operator/=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<double> & (elsa::DataContainer<double>::*) (double) )(
                 &elsa::DataContainer<double>::operator=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("viewAs",
             (elsa::DataContainer<double>(elsa::DataContainer<double>::*)(
                 const elsa::DataDescriptor&))(&elsa::DataContainer<double>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (elsa::DataContainer<double>(elsa::DataContainer<double>::*)(long))(
                 &elsa::DataContainer<double>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (elsa::DataContainer<double>(elsa::DataContainer<double>::*)(long))(
                 &elsa::DataContainer<double>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("loadToCPU",
             (elsa::DataContainer<double>(elsa::DataContainer<double>::*)())(
                 &elsa::DataContainer<double>::loadToCPU),
             py::return_value_policy::move)
        .def("loadToGPU",
             (elsa::DataContainer<double>(elsa::DataContainer<double>::*)())(
                 &elsa::DataContainer<double>::loadToGPU),
             py::return_value_policy::move)
        .def("viewAs",
             (const elsa::DataContainer<double> (elsa::DataContainer<double>::*)(
                 const elsa::DataDescriptor&) const)(&elsa::DataContainer<double>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (const elsa::DataContainer<double> (elsa::DataContainer<double>::*)(long)
                  const)(&elsa::DataContainer<double>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (const elsa::DataContainer<double> (elsa::DataContainer<double>::*)(long)
                  const)(&elsa::DataContainer<double>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("getDataDescriptor",
             (const elsa::DataDescriptor& (elsa::DataContainer<double>::*) ()
                  const)(&elsa::DataContainer<double>::getDataDescriptor),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (const double& (elsa::DataContainer<double>::*) (long)
                  const)(&elsa::DataContainer<double>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (double& (elsa::DataContainer<double>::*) (long) )(
                 &elsa::DataContainer<double>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("at",
             (double(elsa::DataContainer<double>::*)(const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataContainer<double>::at),
             py::arg("coordinate"))
        .def("dot",
             (double(elsa::DataContainer<double>::*)(const elsa::DataContainer<double>&)
                  const)(&elsa::DataContainer<double>::dot),
             py::arg("other"))
        .def("l1Norm",
             (double(elsa::DataContainer<double>::*)() const)(&elsa::DataContainer<double>::l1Norm))
        .def("l2Norm",
             (double(elsa::DataContainer<double>::*)() const)(&elsa::DataContainer<double>::l2Norm))
        .def("lInfNorm", (double(elsa::DataContainer<double>::*)()
                              const)(&elsa::DataContainer<double>::lInfNorm))
        .def("maxElement", (double(elsa::DataContainer<double>::*)()
                                const)(&elsa::DataContainer<double>::maxElement))
        .def("minElement", (double(elsa::DataContainer<double>::*)()
                                const)(&elsa::DataContainer<double>::minElement))
        .def("squaredL2Norm", (double(elsa::DataContainer<double>::*)()
                                   const)(&elsa::DataContainer<double>::squaredL2Norm))
        .def("sum",
             (double(elsa::DataContainer<double>::*)() const)(&elsa::DataContainer<double>::sum))
        .def("getDataHandlerType",
             (elsa::DataHandlerType(elsa::DataContainer<double>::*)()
                  const)(&elsa::DataContainer<double>::getDataHandlerType),
             py::return_value_policy::move)
        .def("getSize",
             (long(elsa::DataContainer<double>::*)() const)(&elsa::DataContainer<double>::getSize))
        .def("l0PseudoNorm", (long(elsa::DataContainer<double>::*)()
                                  const)(&elsa::DataContainer<double>::l0PseudoNorm))
        .def(
            "format",
            (void(elsa::DataContainer<double>::*)(std::basic_ostream<char, std::char_traits<char>>&)
                 const)(&elsa::DataContainer<double>::format),
            py::arg("os"))
        .def("format",
             (void(elsa::DataContainer<double>::*)(
                 std::basic_ostream<char, std::char_traits<char>>&, elsa::format_config)
                  const)(&elsa::DataContainer<double>::format),
             py::arg("os"), py::arg("cfg"))
        .def(py::init<const elsa::DataContainer<double>&>(), py::arg("other"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<double, -1, 1, 0, -1, 1>&,
                      elsa::DataHandlerType>(),
             py::arg("dataDescriptor"), py::arg("data"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def(py::init<const elsa::DataDescriptor&, elsa::DataHandlerType>(),
             py::arg("dataDescriptor"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def("fft",
             (void(elsa::DataContainer<double>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<double>::fft),
             py::arg("norm"))
        .def("ifft",
             (void(elsa::DataContainer<double>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<double>::ifft),
             py::arg("norm"));

    elsa::DataContainerHints<double>::addCustomMethods(DataContainerd);

    elsa::DataContainerHints<double>::exposeBufferInfo(DataContainerd);

    py::class_<elsa::DataContainer<std::complex<double>>> DataContainercd(m, "DataContainercd");
    DataContainercd
        .def("__ne__",
             (bool(elsa::DataContainer<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::DataContainer<std::complex<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::DataContainer<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::DataContainer<std::complex<double>>::operator==),
             py::arg("other"))
        .def(
            "__imul__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (std::complex<double>) )(
                &elsa::DataContainer<std::complex<double>>::operator*=),
            py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "__iadd__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (std::complex<double>) )(
                &elsa::DataContainer<std::complex<double>>::operator+=),
            py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "__isub__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (std::complex<double>) )(
                &elsa::DataContainer<std::complex<double>>::operator-=),
            py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "__idiv__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (std::complex<double>) )(
                &elsa::DataContainer<std::complex<double>>::operator/=),
            py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (std::complex<double>) )(
                &elsa::DataContainer<std::complex<double>>::operator=),
            py::arg("scalar"), py::return_value_policy::reference_internal)
        .def(
            "__imul__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (const elsa::DataContainer<std::complex<double>>&) )(
                &elsa::DataContainer<std::complex<double>>::operator*=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__iadd__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (const elsa::DataContainer<std::complex<double>>&) )(
                &elsa::DataContainer<std::complex<double>>::operator+=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__isub__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (const elsa::DataContainer<std::complex<double>>&) )(
                &elsa::DataContainer<std::complex<double>>::operator-=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "__idiv__",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (const elsa::DataContainer<std::complex<double>>&) )(
                &elsa::DataContainer<std::complex<double>>::operator/=),
            py::arg("dc"), py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::DataContainer<std::complex<
                 double>> & (elsa::DataContainer<std::complex<double>>::*) (const elsa::DataContainer<std::complex<double>>&) )(
                &elsa::DataContainer<std::complex<double>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("viewAs",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::DataContainer<std::complex<double>>::*)(const elsa::DataDescriptor&))(
                 &elsa::DataContainer<std::complex<double>>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::DataContainer<std::complex<double>>::*)(long))(
                 &elsa::DataContainer<std::complex<double>>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::DataContainer<std::complex<double>>::*)(long))(
                 &elsa::DataContainer<std::complex<double>>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("loadToCPU",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::DataContainer<std::complex<double>>::*)())(
                 &elsa::DataContainer<std::complex<double>>::loadToCPU),
             py::return_value_policy::move)
        .def("loadToGPU",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::DataContainer<std::complex<double>>::*)())(
                 &elsa::DataContainer<std::complex<double>>::loadToGPU),
             py::return_value_policy::move)
        .def("__getitem__",
             (std::complex<double> & (elsa::DataContainer<std::complex<double>>::*) (long) )(
                 &elsa::DataContainer<std::complex<double>>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("at",
             (std::complex<double>(elsa::DataContainer<std::complex<double>>::*)(
                 const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataContainer<std::complex<double>>::at),
             py::arg("coordinate"), py::return_value_policy::move)
        .def("dot",
             (std::complex<double>(elsa::DataContainer<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::DataContainer<std::complex<double>>::dot),
             py::arg("other"), py::return_value_policy::move)
        .def("maxElement",
             (std::complex<double>(elsa::DataContainer<std::complex<double>>::*)()
                  const)(&elsa::DataContainer<std::complex<double>>::maxElement),
             py::return_value_policy::move)
        .def("minElement",
             (std::complex<double>(elsa::DataContainer<std::complex<double>>::*)()
                  const)(&elsa::DataContainer<std::complex<double>>::minElement),
             py::return_value_policy::move)
        .def("sum",
             (std::complex<double>(elsa::DataContainer<std::complex<double>>::*)()
                  const)(&elsa::DataContainer<std::complex<double>>::sum),
             py::return_value_policy::move)
        .def("viewAs",
             (const elsa::DataContainer<std::complex<double>> (
                 elsa::DataContainer<std::complex<double>>::*)(const elsa::DataDescriptor&)
                  const)(&elsa::DataContainer<std::complex<double>>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (const elsa::DataContainer<std::complex<double>> (
                 elsa::DataContainer<std::complex<double>>::*)(long)
                  const)(&elsa::DataContainer<std::complex<double>>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (const elsa::DataContainer<std::complex<double>> (
                 elsa::DataContainer<std::complex<double>>::*)(long)
                  const)(&elsa::DataContainer<std::complex<double>>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("getDataDescriptor",
             (const elsa::DataDescriptor& (elsa::DataContainer<std::complex<double>>::*) ()
                  const)(&elsa::DataContainer<std::complex<double>>::getDataDescriptor),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (const std::complex<double>& (elsa::DataContainer<std::complex<double>>::*) (long)
                  const)(&elsa::DataContainer<std::complex<double>>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("l1Norm", (double(elsa::DataContainer<std::complex<double>>::*)()
                            const)(&elsa::DataContainer<std::complex<double>>::l1Norm))
        .def("l2Norm", (double(elsa::DataContainer<std::complex<double>>::*)()
                            const)(&elsa::DataContainer<std::complex<double>>::l2Norm))
        .def("lInfNorm", (double(elsa::DataContainer<std::complex<double>>::*)()
                              const)(&elsa::DataContainer<std::complex<double>>::lInfNorm))
        .def("squaredL2Norm", (double(elsa::DataContainer<std::complex<double>>::*)() const)(
                                  &elsa::DataContainer<std::complex<double>>::squaredL2Norm))
        .def("getDataHandlerType",
             (elsa::DataHandlerType(elsa::DataContainer<std::complex<double>>::*)()
                  const)(&elsa::DataContainer<std::complex<double>>::getDataHandlerType),
             py::return_value_policy::move)
        .def("getSize", (long(elsa::DataContainer<std::complex<double>>::*)()
                             const)(&elsa::DataContainer<std::complex<double>>::getSize))
        .def("l0PseudoNorm", (long(elsa::DataContainer<std::complex<double>>::*)()
                                  const)(&elsa::DataContainer<std::complex<double>>::l0PseudoNorm))
        .def("format",
             (void(elsa::DataContainer<std::complex<double>>::*)(
                 std::basic_ostream<char, std::char_traits<char>>&)
                  const)(&elsa::DataContainer<std::complex<double>>::format),
             py::arg("os"))
        .def("format",
             (void(elsa::DataContainer<std::complex<double>>::*)(
                 std::basic_ostream<char, std::char_traits<char>>&, elsa::format_config)
                  const)(&elsa::DataContainer<std::complex<double>>::format),
             py::arg("os"), py::arg("cfg"))
        .def(py::init<const elsa::DataContainer<std::complex<double>>&>(), py::arg("other"))
        .def(py::init<const elsa::DataDescriptor&,
                      const Eigen::Matrix<std::complex<double>, -1, 1, 0, -1, 1>&,
                      elsa::DataHandlerType>(),
             py::arg("dataDescriptor"), py::arg("data"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def(py::init<const elsa::DataDescriptor&, elsa::DataHandlerType>(),
             py::arg("dataDescriptor"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def("fft",
             (void(elsa::DataContainer<std::complex<double>>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<std::complex<double>>::fft),
             py::arg("norm"))
        .def("ifft",
             (void(elsa::DataContainer<std::complex<double>>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<std::complex<double>>::ifft),
             py::arg("norm"));

    elsa::DataContainerComplexHints<std::complex<double>>::addCustomMethods(DataContainercd);

    py::class_<elsa::DataContainer<long>> DataContainerl(m, "DataContainerl",
                                                         py::buffer_protocol());
    DataContainerl
        .def("__ne__",
             (bool(elsa::DataContainer<long>::*)(const elsa::DataContainer<long>&)
                  const)(&elsa::DataContainer<long>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::DataContainer<long>::*)(const elsa::DataContainer<long>&)
                  const)(&elsa::DataContainer<long>::operator==),
             py::arg("other"))
        .def("__imul__",
             (elsa::DataContainer<
                  long> & (elsa::DataContainer<long>::*) (const elsa::DataContainer<long>&) )(
                 &elsa::DataContainer<long>::operator*=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<
                  long> & (elsa::DataContainer<long>::*) (const elsa::DataContainer<long>&) )(
                 &elsa::DataContainer<long>::operator+=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<
                  long> & (elsa::DataContainer<long>::*) (const elsa::DataContainer<long>&) )(
                 &elsa::DataContainer<long>::operator-=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<
                  long> & (elsa::DataContainer<long>::*) (const elsa::DataContainer<long>&) )(
                 &elsa::DataContainer<long>::operator/=),
             py::arg("dc"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<
                  long> & (elsa::DataContainer<long>::*) (const elsa::DataContainer<long>&) )(
                 &elsa::DataContainer<long>::operator=),
             py::arg("other"), py::return_value_policy::reference_internal)
        .def("__imul__",
             (elsa::DataContainer<long> & (elsa::DataContainer<long>::*) (long) )(
                 &elsa::DataContainer<long>::operator*=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__iadd__",
             (elsa::DataContainer<long> & (elsa::DataContainer<long>::*) (long) )(
                 &elsa::DataContainer<long>::operator+=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__isub__",
             (elsa::DataContainer<long> & (elsa::DataContainer<long>::*) (long) )(
                 &elsa::DataContainer<long>::operator-=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("__idiv__",
             (elsa::DataContainer<long> & (elsa::DataContainer<long>::*) (long) )(
                 &elsa::DataContainer<long>::operator/=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("set",
             (elsa::DataContainer<long> & (elsa::DataContainer<long>::*) (long) )(
                 &elsa::DataContainer<long>::operator=),
             py::arg("scalar"), py::return_value_policy::reference_internal)
        .def("viewAs",
             (elsa::DataContainer<long>(elsa::DataContainer<long>::*)(const elsa::DataDescriptor&))(
                 &elsa::DataContainer<long>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (elsa::DataContainer<long>(elsa::DataContainer<long>::*)(long))(
                 &elsa::DataContainer<long>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (elsa::DataContainer<long>(elsa::DataContainer<long>::*)(long))(
                 &elsa::DataContainer<long>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("loadToCPU",
             (elsa::DataContainer<long>(elsa::DataContainer<long>::*)())(
                 &elsa::DataContainer<long>::loadToCPU),
             py::return_value_policy::move)
        .def("loadToGPU",
             (elsa::DataContainer<long>(elsa::DataContainer<long>::*)())(
                 &elsa::DataContainer<long>::loadToGPU),
             py::return_value_policy::move)
        .def("viewAs",
             (const elsa::DataContainer<long> (elsa::DataContainer<long>::*)(
                 const elsa::DataDescriptor&) const)(&elsa::DataContainer<long>::viewAs),
             py::arg("dataDescriptor"), py::return_value_policy::move)
        .def("getBlock",
             (const elsa::DataContainer<long> (elsa::DataContainer<long>::*)(long)
                  const)(&elsa::DataContainer<long>::getBlock),
             py::arg("i"), py::return_value_policy::move)
        .def("slice",
             (const elsa::DataContainer<long> (elsa::DataContainer<long>::*)(long)
                  const)(&elsa::DataContainer<long>::slice),
             py::arg("i"), py::return_value_policy::move)
        .def("getDataDescriptor",
             (const elsa::DataDescriptor& (elsa::DataContainer<long>::*) ()
                  const)(&elsa::DataContainer<long>::getDataDescriptor),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (const long& (elsa::DataContainer<long>::*) (long)
                  const)(&elsa::DataContainer<long>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("getDataHandlerType",
             (elsa::DataHandlerType(elsa::DataContainer<long>::*)()
                  const)(&elsa::DataContainer<long>::getDataHandlerType),
             py::return_value_policy::move)
        .def("__getitem__",
             (long& (elsa::DataContainer<long>::*) (long) )(&elsa::DataContainer<long>::operator[]),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("at",
             (long(elsa::DataContainer<long>::*)(const Eigen::Matrix<long, -1, 1, 0, -1, 1>&)
                  const)(&elsa::DataContainer<long>::at),
             py::arg("coordinate"))
        .def("dot",
             (long(elsa::DataContainer<long>::*)(const elsa::DataContainer<long>&)
                  const)(&elsa::DataContainer<long>::dot),
             py::arg("other"))
        .def("getSize",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::getSize))
        .def("l0PseudoNorm",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::l0PseudoNorm))
        .def("l1Norm",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::l1Norm))
        .def("l2Norm",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::l2Norm))
        .def("lInfNorm",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::lInfNorm))
        .def("maxElement",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::maxElement))
        .def("minElement",
             (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::minElement))
        .def("squaredL2Norm", (long(elsa::DataContainer<long>::*)()
                                   const)(&elsa::DataContainer<long>::squaredL2Norm))
        .def("sum", (long(elsa::DataContainer<long>::*)() const)(&elsa::DataContainer<long>::sum))
        .def("format",
             (void(elsa::DataContainer<long>::*)(std::basic_ostream<char, std::char_traits<char>>&)
                  const)(&elsa::DataContainer<long>::format),
             py::arg("os"))
        .def("format",
             (void(elsa::DataContainer<long>::*)(std::basic_ostream<char, std::char_traits<char>>&,
                                                 elsa::format_config)
                  const)(&elsa::DataContainer<long>::format),
             py::arg("os"), py::arg("cfg"))
        .def(py::init<const elsa::DataContainer<long>&>(), py::arg("other"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<long, -1, 1, 0, -1, 1>&,
                      elsa::DataHandlerType>(),
             py::arg("dataDescriptor"), py::arg("data"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def(py::init<const elsa::DataDescriptor&, elsa::DataHandlerType>(),
             py::arg("dataDescriptor"),
             py::arg("handlerType") = static_cast<elsa::DataHandlerType>(0))
        .def("fft",
             (void(elsa::DataContainer<long>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<long>::fft),
             py::arg("norm"))
        .def("ifft",
             (void(elsa::DataContainer<long>::*)(elsa::FFTNorm)
                  const)(&elsa::DataContainer<long>::ifft),
             py::arg("norm"));

    elsa::DataContainerHints<long>::addCustomMethods(DataContainerl);

    elsa::DataContainerHints<long>::exposeBufferInfo(DataContainerl);

    py::class_<elsa::Cloneable<elsa::LinearOperator<float>>> CloneableLinearOperatorf(
        m, "CloneableLinearOperatorf");
    CloneableLinearOperatorf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::LinearOperator<float>>::*)(
                 const elsa::LinearOperator<float>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::LinearOperator<float>>::*)(
                 const elsa::LinearOperator<float>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<float>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::LinearOperator<float>,
                                       std::default_delete<elsa::LinearOperator<float>>>(
                          elsa::Cloneable<elsa::LinearOperator<float>>::*)()
                           const)(&elsa::Cloneable<elsa::LinearOperator<float>>::clone));

    py::class_<elsa::LinearOperator<float>, elsa::Cloneable<elsa::LinearOperator<float>>>
        LinearOperatorf(m, "LinearOperatorf");
    LinearOperatorf
        .def("apply",
             (elsa::DataContainer<float>(elsa::LinearOperator<float>::*)(
                 const elsa::DataContainer<float>&) const)(&elsa::LinearOperator<float>::apply),
             py::arg("x"), py::return_value_policy::move)
        .def("applyAdjoint",
             (elsa::DataContainer<float>(elsa::LinearOperator<float>::*)(
                 const elsa::DataContainer<float>&)
                  const)(&elsa::LinearOperator<float>::applyAdjoint),
             py::arg("y"), py::return_value_policy::move)
        .def("set",
             (elsa::LinearOperator<
                  float> & (elsa::LinearOperator<float>::*) (const elsa::LinearOperator<float>&) )(
                 &elsa::LinearOperator<float>::operator=),
             py::arg("other"), py::return_value_policy::reference_internal)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<float>::*) ()
                  const)(&elsa::LinearOperator<float>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<float>::*) ()
                  const)(&elsa::LinearOperator<float>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::LinearOperator<float>::*)(const elsa::DataContainer<float>&,
                                                   elsa::DataContainer<float>&)
                  const)(&elsa::LinearOperator<float>::apply),
             py::arg("x"), py::arg("Ax"))
        .def("applyAdjoint",
             (void(elsa::LinearOperator<float>::*)(const elsa::DataContainer<float>&,
                                                   elsa::DataContainer<float>&)
                  const)(&elsa::LinearOperator<float>::applyAdjoint),
             py::arg("y"), py::arg("Aty"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
             py::arg("domainDescriptor"), py::arg("rangeDescriptor"))
        .def(py::init<const elsa::LinearOperator<float>&>(), py::arg("other"));

    elsa::LinearOperatorHints<float>::addCustomMethods(LinearOperatorf);

    m.attr("LinearOperator") = m.attr("LinearOperatorf");

    py::class_<elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>>
        CloneableLinearOperatorcf(m, "CloneableLinearOperatorcf");
    CloneableLinearOperatorcf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::*)(
                 const elsa::LinearOperator<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::*)(
                 const elsa::LinearOperator<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::LinearOperator<std::complex<float>>,
                              std::default_delete<elsa::LinearOperator<std::complex<float>>>>(
                 elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::*)()
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>::clone));

    py::class_<elsa::LinearOperator<std::complex<float>>,
               elsa::Cloneable<elsa::LinearOperator<std::complex<float>>>>
        LinearOperatorcf(m, "LinearOperatorcf");
    LinearOperatorcf
        .def(
            "apply",
            (elsa::DataContainer<std::complex<float>>(elsa::LinearOperator<std::complex<float>>::*)(
                const elsa::DataContainer<std::complex<float>>&)
                 const)(&elsa::LinearOperator<std::complex<float>>::apply),
            py::arg("x"), py::return_value_policy::move)
        .def(
            "applyAdjoint",
            (elsa::DataContainer<std::complex<float>>(elsa::LinearOperator<std::complex<float>>::*)(
                const elsa::DataContainer<std::complex<float>>&)
                 const)(&elsa::LinearOperator<std::complex<float>>::applyAdjoint),
            py::arg("y"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::LinearOperator<std::complex<
                 float>> & (elsa::LinearOperator<std::complex<float>>::*) (const elsa::LinearOperator<std::complex<float>>&) )(
                &elsa::LinearOperator<std::complex<float>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<std::complex<float>>::*) ()
                  const)(&elsa::LinearOperator<std::complex<float>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<std::complex<float>>::*) ()
                  const)(&elsa::LinearOperator<std::complex<float>>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::LinearOperator<std::complex<float>>::*)(
                 const elsa::DataContainer<std::complex<float>>&,
                 elsa::DataContainer<std::complex<float>>&)
                  const)(&elsa::LinearOperator<std::complex<float>>::apply),
             py::arg("x"), py::arg("Ax"))
        .def("applyAdjoint",
             (void(elsa::LinearOperator<std::complex<float>>::*)(
                 const elsa::DataContainer<std::complex<float>>&,
                 elsa::DataContainer<std::complex<float>>&)
                  const)(&elsa::LinearOperator<std::complex<float>>::applyAdjoint),
             py::arg("y"), py::arg("Aty"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
             py::arg("domainDescriptor"), py::arg("rangeDescriptor"))
        .def(py::init<const elsa::LinearOperator<std::complex<float>>&>(), py::arg("other"));

    elsa::LinearOperatorHints<std::complex<float>>::addCustomMethods(LinearOperatorcf);

    py::class_<elsa::Cloneable<elsa::LinearOperator<double>>> CloneableLinearOperatord(
        m, "CloneableLinearOperatord");
    CloneableLinearOperatord
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::LinearOperator<double>>::*)(
                 const elsa::LinearOperator<double>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::LinearOperator<double>>::*)(
                 const elsa::LinearOperator<double>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<double>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::LinearOperator<double>,
                                       std::default_delete<elsa::LinearOperator<double>>>(
                          elsa::Cloneable<elsa::LinearOperator<double>>::*)()
                           const)(&elsa::Cloneable<elsa::LinearOperator<double>>::clone));

    py::class_<elsa::LinearOperator<double>, elsa::Cloneable<elsa::LinearOperator<double>>>
        LinearOperatord(m, "LinearOperatord");
    LinearOperatord
        .def("apply",
             (elsa::DataContainer<double>(elsa::LinearOperator<double>::*)(
                 const elsa::DataContainer<double>&) const)(&elsa::LinearOperator<double>::apply),
             py::arg("x"), py::return_value_policy::move)
        .def("applyAdjoint",
             (elsa::DataContainer<double>(elsa::LinearOperator<double>::*)(
                 const elsa::DataContainer<double>&)
                  const)(&elsa::LinearOperator<double>::applyAdjoint),
             py::arg("y"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::LinearOperator<
                 double> & (elsa::LinearOperator<double>::*) (const elsa::LinearOperator<double>&) )(
                &elsa::LinearOperator<double>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<double>::*) ()
                  const)(&elsa::LinearOperator<double>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<double>::*) ()
                  const)(&elsa::LinearOperator<double>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::LinearOperator<double>::*)(const elsa::DataContainer<double>&,
                                                    elsa::DataContainer<double>&)
                  const)(&elsa::LinearOperator<double>::apply),
             py::arg("x"), py::arg("Ax"))
        .def("applyAdjoint",
             (void(elsa::LinearOperator<double>::*)(const elsa::DataContainer<double>&,
                                                    elsa::DataContainer<double>&)
                  const)(&elsa::LinearOperator<double>::applyAdjoint),
             py::arg("y"), py::arg("Aty"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
             py::arg("domainDescriptor"), py::arg("rangeDescriptor"))
        .def(py::init<const elsa::LinearOperator<double>&>(), py::arg("other"));

    elsa::LinearOperatorHints<double>::addCustomMethods(LinearOperatord);

    py::class_<elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>>
        CloneableLinearOperatorcd(m, "CloneableLinearOperatorcd");
    CloneableLinearOperatorcd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::*)(
                 const elsa::LinearOperator<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::*)(
                 const elsa::LinearOperator<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::LinearOperator<std::complex<double>>,
                              std::default_delete<elsa::LinearOperator<std::complex<double>>>>(
                 elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::*)()
                  const)(&elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>::clone));

    py::class_<elsa::LinearOperator<std::complex<double>>,
               elsa::Cloneable<elsa::LinearOperator<std::complex<double>>>>
        LinearOperatorcd(m, "LinearOperatorcd");
    LinearOperatorcd
        .def("apply",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::LinearOperator<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::LinearOperator<std::complex<double>>::apply),
             py::arg("x"), py::return_value_policy::move)
        .def("applyAdjoint",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::LinearOperator<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::LinearOperator<std::complex<double>>::applyAdjoint),
             py::arg("y"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::LinearOperator<std::complex<
                 double>> & (elsa::LinearOperator<std::complex<double>>::*) (const elsa::LinearOperator<std::complex<double>>&) )(
                &elsa::LinearOperator<std::complex<double>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<std::complex<double>>::*) ()
                  const)(&elsa::LinearOperator<std::complex<double>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::LinearOperator<std::complex<double>>::*) ()
                  const)(&elsa::LinearOperator<std::complex<double>>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::LinearOperator<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&,
                 elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::LinearOperator<std::complex<double>>::apply),
             py::arg("x"), py::arg("Ax"))
        .def("applyAdjoint",
             (void(elsa::LinearOperator<std::complex<double>>::*)(
                 const elsa::DataContainer<std::complex<double>>&,
                 elsa::DataContainer<std::complex<double>>&)
                  const)(&elsa::LinearOperator<std::complex<double>>::applyAdjoint),
             py::arg("y"), py::arg("Aty"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
             py::arg("domainDescriptor"), py::arg("rangeDescriptor"))
        .def(py::init<const elsa::LinearOperator<std::complex<double>>&>(), py::arg("other"));

    elsa::LinearOperatorHints<std::complex<double>>::addCustomMethods(LinearOperatorcd);

    py::class_<elsa::VolumeDescriptor, elsa::DataDescriptor> VolumeDescriptor(m,
                                                                              "VolumeDescriptor");
    VolumeDescriptor
        .def(py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>>(),
             py::arg("numberOfCoefficientsPerDimension"))
        .def(
            py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
            py::arg("numberOfCoefficientsPerDimension"), py::arg("spacingPerDimension"))
        .def(py::init<std::initializer_list<long>>(), py::arg("numberOfCoefficientsPerDimension"))
        .def(py::init<std::initializer_list<long>, std::initializer_list<float>>(),
             py::arg("numberOfCoefficientsPerDimension"), py::arg("spacingPerDimension"))
        .def(py::init<const elsa::VolumeDescriptor&>());

    py::class_<elsa::BlockDescriptor, elsa::DataDescriptor> BlockDescriptor(m, "BlockDescriptor");
    BlockDescriptor
        .def("getDescriptorOfBlock",
             (const elsa::DataDescriptor& (elsa::BlockDescriptor::*) (long)
                  const)(&elsa::BlockDescriptor::getDescriptorOfBlock),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock",
             (long(elsa::BlockDescriptor::*)(long) const)(&elsa::BlockDescriptor::getOffsetOfBlock),
             py::arg("i"))
        .def("getNumberOfBlocks",
             (long(elsa::BlockDescriptor::*)() const)(&elsa::BlockDescriptor::getNumberOfBlocks));

    py::class_<elsa::IdenticalBlocksDescriptor, elsa::BlockDescriptor> IdenticalBlocksDescriptor(
        m, "IdenticalBlocksDescriptor");
    IdenticalBlocksDescriptor
        .def("getDescriptorOfBlock",
             (const elsa::DataDescriptor& (elsa::IdenticalBlocksDescriptor::*) (long)
                  const)(&elsa::IdenticalBlocksDescriptor::getDescriptorOfBlock),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock",
             (long(elsa::IdenticalBlocksDescriptor::*)(long)
                  const)(&elsa::IdenticalBlocksDescriptor::getOffsetOfBlock),
             py::arg("i"))
        .def("getNumberOfBlocks", (long(elsa::IdenticalBlocksDescriptor::*)()
                                       const)(&elsa::IdenticalBlocksDescriptor::getNumberOfBlocks))
        .def(py::init<long, const elsa::DataDescriptor&>(), py::arg("numberOfBlocks"),
             py::arg("dataDescriptor"));

    py::class_<elsa::PartitionDescriptor, elsa::BlockDescriptor> PartitionDescriptor(
        m, "PartitionDescriptor");
    PartitionDescriptor
        .def("getDescriptorOfBlock",
             (const elsa::DataDescriptor& (elsa::PartitionDescriptor::*) (long)
                  const)(&elsa::PartitionDescriptor::getDescriptorOfBlock),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock",
             (long(elsa::PartitionDescriptor::*)(long)
                  const)(&elsa::PartitionDescriptor::getOffsetOfBlock),
             py::arg("i"))
        .def("getNumberOfBlocks", (long(elsa::PartitionDescriptor::*)()
                                       const)(&elsa::PartitionDescriptor::getNumberOfBlocks))
        .def(py::init<const elsa::DataDescriptor&, Eigen::Matrix<long, -1, 1, 0, -1, 1>>(),
             py::arg("dataDescriptor"), py::arg("slicesInBlock"))
        .def(py::init<const elsa::DataDescriptor&, long>(), py::arg("dataDescriptor"),
             py::arg("numberOfBlocks"));

    py::class_<elsa::RandomBlocksDescriptor, elsa::BlockDescriptor> RandomBlocksDescriptor(
        m, "RandomBlocksDescriptor");
    RandomBlocksDescriptor
        .def("getDescriptorOfBlock",
             (const elsa::DataDescriptor& (elsa::RandomBlocksDescriptor::*) (long)
                  const)(&elsa::RandomBlocksDescriptor::getDescriptorOfBlock),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock",
             (long(elsa::RandomBlocksDescriptor::*)(long)
                  const)(&elsa::RandomBlocksDescriptor::getOffsetOfBlock),
             py::arg("i"))
        .def("getNumberOfBlocks", (long(elsa::RandomBlocksDescriptor::*)()
                                       const)(&elsa::RandomBlocksDescriptor::getNumberOfBlocks));

    elsa::RandomBlocksDescriptorHints::addCustomMethods(RandomBlocksDescriptor);

    py::class_<elsa::geometry::detail::RealWrapper> RealWrapper(m, "RealWrapper");
    RealWrapper.def(py::init<const elsa::geometry::detail::RealWrapper&>())
        .def(py::init<float>(), py::arg("x"))
        .def(py::init<>());

    py::class_<elsa::geometry::SourceToCenterOfRotation> SourceToCenterOfRotation(
        m, "SourceToCenterOfRotation");
    SourceToCenterOfRotation.def(py::init<const elsa::geometry::SourceToCenterOfRotation&>())
        .def(py::init<float>(), py::arg("x"))
        .def(py::init<>());

    py::class_<elsa::geometry::CenterOfRotationToDetector> CenterOfRotationToDetector(
        m, "CenterOfRotationToDetector");
    CenterOfRotationToDetector.def(py::init<const elsa::geometry::CenterOfRotationToDetector&>())
        .def(py::init<float>(), py::arg("x"))
        .def(py::init<>());

    py::class_<elsa::geometry::Degree> Degree(m, "Degree");
    Degree
        .def("to_radian",
             (float(elsa::geometry::Degree::*)() const)(&elsa::geometry::Degree::to_radian))
        .def(py::init<elsa::geometry::Radian>(), py::arg("r"))
        .def(py::init<const elsa::geometry::Degree&>())
        .def(py::init<float>(), py::arg("degree"))
        .def(py::init<>());

    py::class_<elsa::geometry::Radian> Radian(m, "Radian");
    Radian
        .def(
            "set",
            (elsa::geometry::Radian & (elsa::geometry::Radian::*) (const elsa::geometry::Radian&) )(
                &elsa::geometry::Radian::operator=),
            py::return_value_policy::reference_internal)
        .def("to_degree",
             (float(elsa::geometry::Radian::*)() const)(&elsa::geometry::Radian::to_degree))
        .def(py::init<elsa::geometry::Degree>(), py::arg("d"))
        .def(py::init<const elsa::geometry::Radian&>())
        .def(py::init<float>(), py::arg("radian"))
        .def(py::init<>());

    py::class_<elsa::geometry::Coefficients<2>> Coefficients2(m, "Coefficients2");
    elsa::TransparentClassHints<elsa::geometry::Coefficients<2>>::addCustomMethods(Coefficients2);

    py::class_<elsa::geometry::Spacing<2>> Spacing2(m, "Spacing2");
    elsa::TransparentClassHints<elsa::geometry::Spacing<2>>::addCustomMethods(Spacing2);

    py::class_<elsa::geometry::OriginShift<2>> OriginShift2(m, "OriginShift2");
    elsa::TransparentClassHints<elsa::geometry::OriginShift<2>>::addCustomMethods(OriginShift2);

    py::class_<elsa::geometry::detail::GeometryData<2>> GeometryData2(m, "GeometryData2");
    GeometryData2
        .def("getLocationOfOrigin",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::geometry::detail::GeometryData<2>::*)()
                  const&) (&elsa::geometry::detail::GeometryData<2>::getLocationOfOrigin),
             py::return_value_policy::move)
        .def("getSpacing",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::geometry::detail::GeometryData<2>::*)()
                  const&) (&elsa::geometry::detail::GeometryData<2>::getSpacing),
             py::return_value_policy::move)
        .def(
            "set",
            (elsa::geometry::detail::GeometryData<
                 2> & (elsa::geometry::detail::GeometryData<2>::*) (const elsa::geometry::detail::GeometryData<2>&) )(
                &elsa::geometry::detail::GeometryData<2>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<2>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<2>, elsa::geometry::Spacing<2>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<2>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<2>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::detail::GeometryData<2>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::VolumeData<2>, elsa::geometry::detail::GeometryData<2>> VolumeData2(
        m, "VolumeData2");
    VolumeData2
        .def("set",
             (elsa::geometry::VolumeData<
                  2> & (elsa::geometry::VolumeData<2>::*) (const elsa::geometry::VolumeData<2>&) )(
                 &elsa::geometry::VolumeData<2>::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<2>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<2>, elsa::geometry::Spacing<2>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<2>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<2>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::VolumeData<2>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::SinogramData<2>, elsa::geometry::detail::GeometryData<2>>
        SinogramData2(m, "SinogramData2");
    SinogramData2
        .def(
            "set",
            (elsa::geometry::SinogramData<
                 2> & (elsa::geometry::SinogramData<2>::*) (const elsa::geometry::SinogramData<2>&) )(
                &elsa::geometry::SinogramData<2>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<2>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<2>, elsa::geometry::Spacing<2>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<2>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<2>, elsa::geometry::OriginShift<2>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::SinogramData<2>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::PrincipalPointOffset> PrincipalPointOffset(m,
                                                                          "PrincipalPointOffset");
    PrincipalPointOffset.def(py::init<const elsa::geometry::PrincipalPointOffset&>())
        .def(py::init<float>(), py::arg("x"))
        .def(py::init<>());

    py::class_<
        elsa::geometry::detail::StaticVectorTemplate<2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>>
        StaticVectorTemplate2Matrixfloat_110_11(m, "StaticVectorTemplate2Matrixfloat_110_11");
    StaticVectorTemplate2Matrixfloat_110_11
        .def(
            "set",
            (elsa::geometry::detail::StaticVectorTemplate<
                 2,
                 Eigen::Matrix<
                     float, -1, 1, 0, -1,
                     1>> & (elsa::geometry::detail::StaticVectorTemplate<2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*) (const elsa::geometry::detail::StaticVectorTemplate<2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&) )(
                &elsa::geometry::detail::StaticVectorTemplate<
                    2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator=),
            py::return_value_policy::reference_internal)
        .def("get",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (
                 elsa::geometry::detail::StaticVectorTemplate<
                     2, Eigen::Matrix<float, -1, 1, 0, -1,
                                      1>>::*) ()&) (&elsa::geometry::detail::
                                                        StaticVectorTemplate<
                                                            2, Eigen::Matrix<float, -1, 1, 0, -1,
                                                                             1>>::get),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float& (elsa::geometry::detail::StaticVectorTemplate<
                      2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*) (long) )(
                 &elsa::geometry::detail::StaticVectorTemplate<
                     2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator[]),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float(elsa::geometry::detail::StaticVectorTemplate<
                    2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*)(long)
                  const)(&elsa::geometry::detail::StaticVectorTemplate<
                         2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator[]),
             py::arg("i"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(), py::arg("vec"))
        .def(py::init<const elsa::geometry::detail::StaticVectorTemplate<
                 2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::RotationOffset<2>> RotationOffset2(m, "RotationOffset2");
    RotationOffset2
        .def(
            "set",
            (elsa::geometry::RotationOffset<
                 2> & (elsa::geometry::RotationOffset<2>::*) (const elsa::geometry::RotationOffset<2>&) )(
                &elsa::geometry::RotationOffset<2>::operator=),
            py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::geometry::detail::StaticVectorTemplate<
                 2,
                 Eigen::Matrix<
                     float, -1, 1, 0, -1,
                     1>> & (elsa::geometry::RotationOffset<2>::*) (const elsa::geometry::detail::StaticVectorTemplate<2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&) )(
                &elsa::geometry::RotationOffset<2>::operator=),
            py::return_value_policy::reference_internal)
        .def("get",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (
                 elsa::geometry::RotationOffset<
                     2>::*) ()&) (&elsa::geometry::RotationOffset<2>::get),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float& (elsa::geometry::RotationOffset<2>::*) (long) )(
                 &elsa::geometry::RotationOffset<2>::operator[]),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float(elsa::geometry::RotationOffset<2>::*)(long)
                  const)(&elsa::geometry::RotationOffset<2>::operator[]),
             py::arg("i"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(), py::arg("vec"))
        .def(py::init<const elsa::geometry::RotationOffset<2>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::Coefficients<3>> Coefficients3(m, "Coefficients3");
    elsa::TransparentClassHints<elsa::geometry::Coefficients<3>>::addCustomMethods(Coefficients3);

    py::class_<elsa::geometry::Spacing<3>> Spacing3(m, "Spacing3");
    elsa::TransparentClassHints<elsa::geometry::Spacing<3>>::addCustomMethods(Spacing3);

    py::class_<elsa::geometry::OriginShift<3>> OriginShift3(m, "OriginShift3");
    elsa::TransparentClassHints<elsa::geometry::OriginShift<3>>::addCustomMethods(OriginShift3);

    py::class_<elsa::geometry::detail::GeometryData<3>> GeometryData3(m, "GeometryData3");
    GeometryData3
        .def("getLocationOfOrigin",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::geometry::detail::GeometryData<3>::*)()
                  const&) (&elsa::geometry::detail::GeometryData<3>::getLocationOfOrigin),
             py::return_value_policy::move)
        .def("getSpacing",
             (Eigen::Matrix<float, -1, 1, 0, -1, 1>(elsa::geometry::detail::GeometryData<3>::*)()
                  const&) (&elsa::geometry::detail::GeometryData<3>::getSpacing),
             py::return_value_policy::move)
        .def(
            "set",
            (elsa::geometry::detail::GeometryData<
                 3> & (elsa::geometry::detail::GeometryData<3>::*) (const elsa::geometry::detail::GeometryData<3>&) )(
                &elsa::geometry::detail::GeometryData<3>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<3>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<3>, elsa::geometry::Spacing<3>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<3>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<3>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::detail::GeometryData<3>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::VolumeData<3>, elsa::geometry::detail::GeometryData<3>> VolumeData3(
        m, "VolumeData3");
    VolumeData3
        .def("set",
             (elsa::geometry::VolumeData<
                  3> & (elsa::geometry::VolumeData<3>::*) (const elsa::geometry::VolumeData<3>&) )(
                 &elsa::geometry::VolumeData<3>::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<3>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<3>, elsa::geometry::Spacing<3>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<3>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<3>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::VolumeData<3>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::SinogramData<3>, elsa::geometry::detail::GeometryData<3>>
        SinogramData3(m, "SinogramData3");
    SinogramData3
        .def(
            "set",
            (elsa::geometry::SinogramData<
                 3> & (elsa::geometry::SinogramData<3>::*) (const elsa::geometry::SinogramData<3>&) )(
                &elsa::geometry::SinogramData<3>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>,
                      Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Coefficients<3>>(), py::arg("size"))
        .def(py::init<elsa::geometry::Coefficients<3>, elsa::geometry::Spacing<3>>(),
             py::arg("size"), py::arg("spacing"))
        .def(py::init<elsa::geometry::Spacing<3>, Eigen::Matrix<float, -1, 1, 0, -1, 1>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<elsa::geometry::Spacing<3>, elsa::geometry::OriginShift<3>>(),
             py::arg("spacing"), py::arg("origin"))
        .def(py::init<const elsa::geometry::SinogramData<3>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::detail::RotationAngles<3>> RotationAngles3(m, "RotationAngles3");
    RotationAngles3
        .def("__getitem__",
             (elsa::geometry::Radian(elsa::geometry::detail::RotationAngles<3>::*)(long)
                  const)(&elsa::geometry::detail::RotationAngles<3>::operator[]),
             py::arg("i"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::geometry::detail::RotationAngles<
                 3> & (elsa::geometry::detail::RotationAngles<3>::*) (const elsa::geometry::detail::RotationAngles<3>&) )(
                &elsa::geometry::detail::RotationAngles<3>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::geometry::detail::RotationAngles<3>&>());

    py::class_<elsa::geometry::Gamma> Gamma(m, "Gamma");
    Gamma
        .def("to_degree",
             (float(elsa::geometry::Gamma::*)() const)(&elsa::geometry::Gamma::to_degree))
        .def(py::init<elsa::geometry::Degree>(), py::arg("d"))
        .def(py::init<const elsa::geometry::Gamma&>())
        .def(py::init<float>(), py::arg("radian"))
        .def(py::init<>());

    py::class_<elsa::geometry::Beta> Beta(m, "Beta");
    Beta.def("to_degree",
             (float(elsa::geometry::Beta::*)() const)(&elsa::geometry::Beta::to_degree))
        .def(py::init<elsa::geometry::Degree>(), py::arg("d"))
        .def(py::init<const elsa::geometry::Beta&>())
        .def(py::init<float>(), py::arg("radian"))
        .def(py::init<>());

    py::class_<elsa::geometry::Alpha> Alpha(m, "Alpha");
    Alpha
        .def("to_degree",
             (float(elsa::geometry::Alpha::*)() const)(&elsa::geometry::Alpha::to_degree))
        .def(py::init<elsa::geometry::Degree>(), py::arg("d"))
        .def(py::init<const elsa::geometry::Alpha&>())
        .def(py::init<float>(), py::arg("radian"))
        .def(py::init<>());

    py::class_<elsa::geometry::RotationAngles3D, elsa::geometry::detail::RotationAngles<3>>
        RotationAngles3D(m, "RotationAngles3D");
    RotationAngles3D
        .def("alpha",
             (elsa::geometry::Radian(elsa::geometry::RotationAngles3D::*)()
                  const)(&elsa::geometry::RotationAngles3D::alpha),
             py::return_value_policy::move)
        .def("beta",
             (elsa::geometry::Radian(elsa::geometry::RotationAngles3D::*)()
                  const)(&elsa::geometry::RotationAngles3D::beta),
             py::return_value_policy::move)
        .def("gamma",
             (elsa::geometry::Radian(elsa::geometry::RotationAngles3D::*)()
                  const)(&elsa::geometry::RotationAngles3D::gamma),
             py::return_value_policy::move)
        .def("set",
             (elsa::geometry::RotationAngles3D
              & (elsa::geometry::RotationAngles3D::*) (const elsa::geometry::RotationAngles3D&) )(
                 &elsa::geometry::RotationAngles3D::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<elsa::geometry::Alpha>(), py::arg("alpha"))
        .def(py::init<elsa::geometry::Alpha, elsa::geometry::Beta>(), py::arg("alpha"),
             py::arg("beta"))
        .def(py::init<elsa::geometry::Alpha, elsa::geometry::Gamma>(), py::arg("alpha"),
             py::arg("gamma"))
        .def(py::init<elsa::geometry::Beta>(), py::arg("beta"))
        .def(py::init<elsa::geometry::Beta, elsa::geometry::Alpha>(), py::arg("beta"),
             py::arg("alpha"))
        .def(py::init<elsa::geometry::Beta, elsa::geometry::Gamma>(), py::arg("beta"),
             py::arg("gamma"))
        .def(py::init<elsa::geometry::Gamma>(), py::arg("gamma"))
        .def(py::init<elsa::geometry::Gamma, elsa::geometry::Alpha>(), py::arg("gamma"),
             py::arg("alpha"))
        .def(py::init<elsa::geometry::Gamma, elsa::geometry::Beta>(), py::arg("gamma"),
             py::arg("beta"))
        .def(py::init<elsa::geometry::Gamma, elsa::geometry::Beta, elsa::geometry::Alpha>(),
             py::arg("gamma"), py::arg("beta"), py::arg("alpha"))
        .def(py::init<const elsa::geometry::RotationAngles3D&>());

    py::class_<elsa::geometry::PrincipalPointOffset2D> PrincipalPointOffset2D(
        m, "PrincipalPointOffset2D");
    PrincipalPointOffset2D
        .def("set",
             (elsa::geometry::PrincipalPointOffset2D
              & (elsa::geometry::PrincipalPointOffset2D::*) (const elsa::geometry::
                                                                 PrincipalPointOffset2D&) )(
                 &elsa::geometry::PrincipalPointOffset2D::operator=),
             py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::geometry::detail::StaticVectorTemplate<
                 2,
                 Eigen::Matrix<
                     float, -1, 1, 0, -1,
                     1>> & (elsa::geometry::PrincipalPointOffset2D::*) (const elsa::geometry::detail::StaticVectorTemplate<2, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&) )(
                &elsa::geometry::PrincipalPointOffset2D::operator=),
            py::return_value_policy::reference_internal)
        .def("get",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (
                 elsa::geometry::PrincipalPointOffset2D::*) ()&) (&elsa::geometry::
                                                                      PrincipalPointOffset2D::get),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float& (elsa::geometry::PrincipalPointOffset2D::*) (long) )(
                 &elsa::geometry::PrincipalPointOffset2D::operator[]),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float(elsa::geometry::PrincipalPointOffset2D::*)(long)
                  const)(&elsa::geometry::PrincipalPointOffset2D::operator[]),
             py::arg("i"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(), py::arg("vec"))
        .def(py::init<const elsa::geometry::PrincipalPointOffset2D&>())
        .def(py::init<>());

    py::class_<
        elsa::geometry::detail::StaticVectorTemplate<3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>>
        StaticVectorTemplate3Matrixfloat_110_11(m, "StaticVectorTemplate3Matrixfloat_110_11");
    StaticVectorTemplate3Matrixfloat_110_11
        .def(
            "set",
            (elsa::geometry::detail::StaticVectorTemplate<
                 3,
                 Eigen::Matrix<
                     float, -1, 1, 0, -1,
                     1>> & (elsa::geometry::detail::StaticVectorTemplate<3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*) (const elsa::geometry::detail::StaticVectorTemplate<3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&) )(
                &elsa::geometry::detail::StaticVectorTemplate<
                    3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator=),
            py::return_value_policy::reference_internal)
        .def("get",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (
                 elsa::geometry::detail::StaticVectorTemplate<
                     3, Eigen::Matrix<float, -1, 1, 0, -1,
                                      1>>::*) ()&) (&elsa::geometry::detail::
                                                        StaticVectorTemplate<
                                                            3, Eigen::Matrix<float, -1, 1, 0, -1,
                                                                             1>>::get),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float& (elsa::geometry::detail::StaticVectorTemplate<
                      3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*) (long) )(
                 &elsa::geometry::detail::StaticVectorTemplate<
                     3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator[]),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float(elsa::geometry::detail::StaticVectorTemplate<
                    3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::*)(long)
                  const)(&elsa::geometry::detail::StaticVectorTemplate<
                         3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>::operator[]),
             py::arg("i"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(), py::arg("vec"))
        .def(py::init<const elsa::geometry::detail::StaticVectorTemplate<
                 3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&>())
        .def(py::init<>());

    py::class_<elsa::geometry::RotationOffset<3>> RotationOffset3(m, "RotationOffset3");
    RotationOffset3
        .def(
            "set",
            (elsa::geometry::RotationOffset<
                 3> & (elsa::geometry::RotationOffset<3>::*) (const elsa::geometry::RotationOffset<3>&) )(
                &elsa::geometry::RotationOffset<3>::operator=),
            py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::geometry::detail::StaticVectorTemplate<
                 3,
                 Eigen::Matrix<
                     float, -1, 1, 0, -1,
                     1>> & (elsa::geometry::RotationOffset<3>::*) (const elsa::geometry::detail::StaticVectorTemplate<3, Eigen::Matrix<float, -1, 1, 0, -1, 1>>&) )(
                &elsa::geometry::RotationOffset<3>::operator=),
            py::return_value_policy::reference_internal)
        .def("get",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (
                 elsa::geometry::RotationOffset<
                     3>::*) ()&) (&elsa::geometry::RotationOffset<3>::get),
             py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float& (elsa::geometry::RotationOffset<3>::*) (long) )(
                 &elsa::geometry::RotationOffset<3>::operator[]),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("__getitem__",
             (float(elsa::geometry::RotationOffset<3>::*)(long)
                  const)(&elsa::geometry::RotationOffset<3>::operator[]),
             py::arg("i"))
        .def(py::init<Eigen::Matrix<float, -1, 1, 0, -1, 1>>(), py::arg("vec"))
        .def(py::init<const elsa::geometry::RotationOffset<3>&>())
        .def(py::init<>());

    py::class_<elsa::Geometry> Geometry(m, "Geometry");
    Geometry
        .def("__eq__",
             (bool(elsa::Geometry::*)(const elsa::Geometry&) const)(&elsa::Geometry::operator==),
             py::arg("other"))
        .def("set",
             (elsa::Geometry
              & (elsa::Geometry::*) (const elsa::Geometry&) )(&elsa::Geometry::operator=),
             py::return_value_policy::reference_internal)
        .def("getInverseProjectionMatrix",
             (const Eigen::Matrix<float, -1, -1, 0, -1, -1>& (elsa::Geometry::*) ()
                  const)(&elsa::Geometry::getInverseProjectionMatrix),
             py::return_value_policy::reference_internal)
        .def("getProjectionMatrix",
             (const Eigen::Matrix<float, -1, -1, 0, -1, -1>& (elsa::Geometry::*) ()
                  const)(&elsa::Geometry::getProjectionMatrix),
             py::return_value_policy::reference_internal)
        .def("getRotationMatrix",
             (const Eigen::Matrix<float, -1, -1, 0, -1, -1>& (elsa::Geometry::*) ()
                  const)(&elsa::Geometry::getRotationMatrix),
             py::return_value_policy::reference_internal)
        .def("getCameraCenter",
             (const Eigen::Matrix<float, -1, 1, 0, -1, 1>& (elsa::Geometry::*) ()
                  const)(&elsa::Geometry::getCameraCenter),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Geometry&>())
        .def(py::init<float, float, const elsa::DataDescriptor&, const elsa::DataDescriptor&,
                      const Eigen::Matrix<float, -1, -1, 0, -1, -1>&, float, float, float, float,
                      float>(),
             py::arg("sourceToCenterOfRotation"), py::arg("centerOfRotationToDetector"),
             py::arg("volumeDescriptor"), py::arg("sinoDescriptor"), py::arg("R"),
             py::arg("px") = static_cast<float>(0.000000e+00),
             py::arg("py") = static_cast<float>(0.000000e+00),
             py::arg("centerOfRotationOffsetX") = static_cast<float>(0.000000e+00),
             py::arg("centerOfRotationOffsetY") = static_cast<float>(0.000000e+00),
             py::arg("centerOfRotationOffsetZ") = static_cast<float>(0.000000e+00));

    elsa::GeometryHints::addCustomMethods(Geometry);

    py::class_<elsa::DetectorDescriptor, elsa::DataDescriptor> DetectorDescriptor(
        m, "DetectorDescriptor");
    DetectorDescriptor
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::DetectorDescriptor::*)(
                 const Eigen::Matrix<long, -1, 1, 0, -1, 1>)
                  const)(&elsa::DetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("coord"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::DetectorDescriptor::*)(
                 const Eigen::Matrix<float, -1, 1, 0, -1, 1>&, const long)
                  const)(&elsa::DetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorCoord"), py::arg("poseIndex"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::DetectorDescriptor::*)(const long)
                  const)(&elsa::DetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorIndex"), py::return_value_policy::move)
        .def("getGeometryAt",
             (std::optional<elsa::Geometry>(elsa::DetectorDescriptor::*)(const long)
                  const)(&elsa::DetectorDescriptor::getGeometryAt),
             py::arg("index"), py::return_value_policy::move)
        .def("getGeometry",
             (std::vector<elsa::Geometry, std::allocator<elsa::Geometry>>(
                 elsa::DetectorDescriptor::*)() const)(&elsa::DetectorDescriptor::getGeometry),
             py::return_value_policy::move)
        .def("getNumberOfGeometryPoses", (long(elsa::DetectorDescriptor::*)() const)(
                                             &elsa::DetectorDescriptor::getNumberOfGeometryPoses));

    py::class_<elsa::PlanarDetectorDescriptor, elsa::DetectorDescriptor> PlanarDetectorDescriptor(
        m, "PlanarDetectorDescriptor");
    PlanarDetectorDescriptor
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::PlanarDetectorDescriptor::*)(
                 const Eigen::Matrix<long, -1, 1, 0, -1, 1>)
                  const)(&elsa::PlanarDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("coord"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::PlanarDetectorDescriptor::*)(
                 const Eigen::Matrix<float, -1, 1, 0, -1, 1>&, const long)
                  const)(&elsa::PlanarDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorCoord"), py::arg("poseIndex"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::PlanarDetectorDescriptor::*)(const long)
                  const)(&elsa::PlanarDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorIndex"), py::return_value_policy::move)
        .def(py::init<const Eigen::Matrix<long, -1, 1, 0, -1, 1>&,
                      const Eigen::Matrix<float, -1, 1, 0, -1, 1>&,
                      const std::vector<elsa::Geometry, std::allocator<elsa::Geometry>>&>(),
             py::arg("numOfCoeffsPerDim"), py::arg("spacingPerDim"), py::arg("geometryList"))
        .def(py::init<const Eigen::Matrix<long, -1, 1, 0, -1, 1>&,
                      const std::vector<elsa::Geometry, std::allocator<elsa::Geometry>>&>(),
             py::arg("numOfCoeffsPerDim"), py::arg("geometryList"))
        .def(py::init<const elsa::PlanarDetectorDescriptor&>());

    py::class_<elsa::CurvedDetectorDescriptor, elsa::DetectorDescriptor> CurvedDetectorDescriptor(
        m, "CurvedDetectorDescriptor");
    CurvedDetectorDescriptor
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::CurvedDetectorDescriptor::*)(
                 const Eigen::Matrix<long, -1, 1, 0, -1, 1>)
                  const)(&elsa::CurvedDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("coord"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::CurvedDetectorDescriptor::*)(
                 const Eigen::Matrix<float, -1, 1, 0, -1, 1>&, const long)
                  const)(&elsa::CurvedDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorCoord"), py::arg("poseIndex"), py::return_value_policy::move)
        .def("computeRayFromDetectorCoord",
             (Eigen::ParametrizedLine<float, -1, 0>(elsa::CurvedDetectorDescriptor::*)(const long)
                  const)(&elsa::CurvedDetectorDescriptor::computeRayFromDetectorCoord),
             py::arg("detectorIndex"), py::return_value_policy::move)
        .def(py::init<const Eigen::Matrix<long, -1, 1, 0, -1, 1>&,
                      const Eigen::Matrix<float, -1, 1, 0, -1, 1>&,
                      const std::vector<elsa::Geometry, std::allocator<elsa::Geometry>>&,
                      elsa::geometry::Radian, float>(),
             py::arg("numOfCoeffsPerDim"), py::arg("spacingPerDim"), py::arg("geometryList"),
             py::arg("angle"), py::arg("s2d"))
        .def(py::init<const Eigen::Matrix<long, -1, 1, 0, -1, 1>&,
                      const std::vector<elsa::Geometry, std::allocator<elsa::Geometry>>&,
                      elsa::geometry::Radian, float>(),
             py::arg("numOfCoeffsPerDim"), py::arg("geometryList"), py::arg("angle"),
             py::arg("s2d"))
        .def(py::init<const elsa::CurvedDetectorDescriptor&>());

    elsa::CoreHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_core, m)
{
    add_definitions_pyelsa_core(m);
}
