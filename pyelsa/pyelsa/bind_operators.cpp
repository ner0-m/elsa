#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "BlockLinearOperator.h"
#include "Dictionary.h"
#include "FiniteDifferences.h"
#include "FourierTransform.h"
#include "Identity.h"
#include "Scaling.h"
#include "ShearletTransform.h"

#include "hints/operators_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_operators(py::module& m)
{
    py::enum_<elsa::FiniteDifferences<float>::DiffType>(m, "FiniteDifferencesfDiffType")
        .value("BACKWARD", elsa::FiniteDifferences<float>::DiffType::BACKWARD)
        .value("CENTRAL", elsa::FiniteDifferences<float>::DiffType::CENTRAL)
        .value("FORWARD", elsa::FiniteDifferences<float>::DiffType::FORWARD);

    py::enum_<elsa::FiniteDifferences<double>::DiffType>(m, "FiniteDifferencesdDiffType")
        .value("BACKWARD", elsa::FiniteDifferences<double>::DiffType::BACKWARD)
        .value("CENTRAL", elsa::FiniteDifferences<double>::DiffType::CENTRAL)
        .value("FORWARD", elsa::FiniteDifferences<double>::DiffType::FORWARD);

    py::enum_<elsa::FiniteDifferences<std::complex<float>>::DiffType>(m,
                                                                      "FiniteDifferencescfDiffType")
        .value("BACKWARD", elsa::FiniteDifferences<std::complex<float>>::DiffType::BACKWARD)
        .value("CENTRAL", elsa::FiniteDifferences<std::complex<float>>::DiffType::CENTRAL)
        .value("FORWARD", elsa::FiniteDifferences<std::complex<float>>::DiffType::FORWARD);

    py::enum_<elsa::FiniteDifferences<std::complex<double>>::DiffType>(
        m, "FiniteDifferencescdDiffType")
        .value("BACKWARD", elsa::FiniteDifferences<std::complex<double>>::DiffType::BACKWARD)
        .value("CENTRAL", elsa::FiniteDifferences<std::complex<double>>::DiffType::CENTRAL)
        .value("FORWARD", elsa::FiniteDifferences<std::complex<double>>::DiffType::FORWARD);

    py::enum_<elsa::BlockLinearOperator<float>::BlockType>(m, "BlockLinearOperatorfBlockType")
        .value("COL", elsa::BlockLinearOperator<float>::BlockType::COL)
        .value("ROW", elsa::BlockLinearOperator<float>::BlockType::ROW)
        .export_values();

    py::enum_<elsa::BlockLinearOperator<double>::BlockType>(m, "BlockLinearOperatordBlockType")
        .value("COL", elsa::BlockLinearOperator<double>::BlockType::COL)
        .value("ROW", elsa::BlockLinearOperator<double>::BlockType::ROW)
        .export_values();

    py::enum_<elsa::BlockLinearOperator<std::complex<float>>::BlockType>(
        m, "BlockLinearOperatorcfBlockType")
        .value("COL", elsa::BlockLinearOperator<std::complex<float>>::BlockType::COL)
        .value("ROW", elsa::BlockLinearOperator<std::complex<float>>::BlockType::ROW)
        .export_values();

    py::enum_<elsa::BlockLinearOperator<std::complex<double>>::BlockType>(
        m, "BlockLinearOperatorcdBlockType")
        .value("COL", elsa::BlockLinearOperator<std::complex<double>>::BlockType::COL)
        .value("ROW", elsa::BlockLinearOperator<std::complex<double>>::BlockType::ROW)
        .export_values();

    py::class_<elsa::Identity<float>, elsa::LinearOperator<float>> Identityf(m, "Identityf");
    Identityf
        .def("set",
             (elsa::Identity<float> & (elsa::Identity<float>::*) (const elsa::Identity<float>&) )(
                 &elsa::Identity<float>::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    m.attr("Identity") = m.attr("Identityf");

    py::class_<elsa::Identity<std::complex<float>>, elsa::LinearOperator<std::complex<float>>>
        Identitycf(m, "Identitycf");
    Identitycf
        .def(
            "set",
            (elsa::Identity<std::complex<
                 float>> & (elsa::Identity<std::complex<float>>::*) (const elsa::Identity<std::complex<float>>&) )(
                &elsa::Identity<std::complex<float>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    py::class_<elsa::Identity<double>, elsa::LinearOperator<double>> Identityd(m, "Identityd");
    Identityd
        .def(
            "set",
            (elsa::Identity<double> & (elsa::Identity<double>::*) (const elsa::Identity<double>&) )(
                &elsa::Identity<double>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    py::class_<elsa::Identity<std::complex<double>>, elsa::LinearOperator<std::complex<double>>>
        Identitycd(m, "Identitycd");
    Identitycd
        .def(
            "set",
            (elsa::Identity<std::complex<
                 double>> & (elsa::Identity<std::complex<double>>::*) (const elsa::Identity<std::complex<double>>&) )(
                &elsa::Identity<std::complex<double>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    py::class_<elsa::Scaling<float>, elsa::LinearOperator<float>> Scalingf(m, "Scalingf");
    Scalingf
        .def("isIsotropic",
             (bool(elsa::Scaling<float>::*)() const)(&elsa::Scaling<float>::isIsotropic))
        .def("getScaleFactors",
             (const elsa::DataContainer<float>& (elsa::Scaling<float>::*) ()
                  const)(&elsa::Scaling<float>::getScaleFactors),
             py::return_value_policy::reference_internal)
        .def("getScaleFactor",
             (float(elsa::Scaling<float>::*)() const)(&elsa::Scaling<float>::getScaleFactor))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<float>&>(),
             py::arg("descriptor"), py::arg("scaleFactors"))
        .def(py::init<const elsa::DataDescriptor&, float>(), py::arg("descriptor"),
             py::arg("scaleFactor"));

    m.attr("Scaling") = m.attr("Scalingf");

    py::class_<elsa::Scaling<std::complex<float>>, elsa::LinearOperator<std::complex<float>>>
        Scalingcf(m, "Scalingcf");
    Scalingcf
        .def("isIsotropic", (bool(elsa::Scaling<std::complex<float>>::*)()
                                 const)(&elsa::Scaling<std::complex<float>>::isIsotropic))
        .def("getScaleFactor",
             (std::complex<float>(elsa::Scaling<std::complex<float>>::*)()
                  const)(&elsa::Scaling<std::complex<float>>::getScaleFactor),
             py::return_value_policy::move)
        .def("getScaleFactors",
             (const elsa::DataContainer<std::complex<float>>& (
                 elsa::Scaling<std::complex<float>>::*) ()
                  const)(&elsa::Scaling<std::complex<float>>::getScaleFactors),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, std::complex<float>>(), py::arg("descriptor"),
             py::arg("scaleFactor"))
        .def(py::init<const elsa::DataDescriptor&,
                      const elsa::DataContainer<std::complex<float>>&>(),
             py::arg("descriptor"), py::arg("scaleFactors"));

    py::class_<elsa::Scaling<double>, elsa::LinearOperator<double>> Scalingd(m, "Scalingd");
    Scalingd
        .def("isIsotropic",
             (bool(elsa::Scaling<double>::*)() const)(&elsa::Scaling<double>::isIsotropic))
        .def("getScaleFactors",
             (const elsa::DataContainer<double>& (elsa::Scaling<double>::*) ()
                  const)(&elsa::Scaling<double>::getScaleFactors),
             py::return_value_policy::reference_internal)
        .def("getScaleFactor",
             (double(elsa::Scaling<double>::*)() const)(&elsa::Scaling<double>::getScaleFactor))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<double>&>(),
             py::arg("descriptor"), py::arg("scaleFactors"))
        .def(py::init<const elsa::DataDescriptor&, double>(), py::arg("descriptor"),
             py::arg("scaleFactor"));

    py::class_<elsa::Scaling<std::complex<double>>, elsa::LinearOperator<std::complex<double>>>
        Scalingcd(m, "Scalingcd");
    Scalingcd
        .def("isIsotropic", (bool(elsa::Scaling<std::complex<double>>::*)()
                                 const)(&elsa::Scaling<std::complex<double>>::isIsotropic))
        .def("getScaleFactor",
             (std::complex<double>(elsa::Scaling<std::complex<double>>::*)()
                  const)(&elsa::Scaling<std::complex<double>>::getScaleFactor),
             py::return_value_policy::move)
        .def("getScaleFactors",
             (const elsa::DataContainer<std::complex<double>>& (
                 elsa::Scaling<std::complex<double>>::*) ()
                  const)(&elsa::Scaling<std::complex<double>>::getScaleFactors),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, std::complex<double>>(), py::arg("descriptor"),
             py::arg("scaleFactor"))
        .def(py::init<const elsa::DataDescriptor&,
                      const elsa::DataContainer<std::complex<double>>&>(),
             py::arg("descriptor"), py::arg("scaleFactors"));

    py::class_<elsa::FiniteDifferences<float>, elsa::LinearOperator<float>> FiniteDifferencesf(
        m, "FiniteDifferencesf");
    FiniteDifferencesf
        .def(
            "set",
            (elsa::FiniteDifferences<
                 float> & (elsa::FiniteDifferences<float>::*) (const elsa::FiniteDifferences<float>&) )(
                &elsa::FiniteDifferences<float>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&>(),
             py::arg("domainDescriptor"), py::arg("activeDims"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&,
                      elsa::FiniteDifferences<float>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("activeDims"), py::arg("type"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::DataDescriptor&, elsa::FiniteDifferences<float>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("type"));

    m.attr("FiniteDifferences") = m.attr("FiniteDifferencesf");

    py::class_<elsa::FiniteDifferences<double>, elsa::LinearOperator<double>> FiniteDifferencesd(
        m, "FiniteDifferencesd");
    FiniteDifferencesd
        .def(
            "set",
            (elsa::FiniteDifferences<
                 double> & (elsa::FiniteDifferences<double>::*) (const elsa::FiniteDifferences<double>&) )(
                &elsa::FiniteDifferences<double>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&>(),
             py::arg("domainDescriptor"), py::arg("activeDims"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&,
                      elsa::FiniteDifferences<double>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("activeDims"), py::arg("type"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::DataDescriptor&, elsa::FiniteDifferences<double>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("type"));

    py::class_<elsa::FiniteDifferences<std::complex<float>>,
               elsa::LinearOperator<std::complex<float>>>
        FiniteDifferencescf(m, "FiniteDifferencescf");
    FiniteDifferencescf
        .def(
            "set",
            (elsa::FiniteDifferences<std::complex<
                 float>> & (elsa::FiniteDifferences<std::complex<float>>::*) (const elsa::FiniteDifferences<std::complex<float>>&) )(
                &elsa::FiniteDifferences<std::complex<float>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&>(),
             py::arg("domainDescriptor"), py::arg("activeDims"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&,
                      elsa::FiniteDifferences<std::complex<float>>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("activeDims"), py::arg("type"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::DataDescriptor&,
                      elsa::FiniteDifferences<std::complex<float>>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("type"));

    py::class_<elsa::FiniteDifferences<std::complex<double>>,
               elsa::LinearOperator<std::complex<double>>>
        FiniteDifferencescd(m, "FiniteDifferencescd");
    FiniteDifferencescd
        .def(
            "set",
            (elsa::FiniteDifferences<std::complex<
                 double>> & (elsa::FiniteDifferences<std::complex<double>>::*) (const elsa::FiniteDifferences<std::complex<double>>&) )(
                &elsa::FiniteDifferences<std::complex<double>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&>(),
             py::arg("domainDescriptor"), py::arg("activeDims"))
        .def(py::init<const elsa::DataDescriptor&, const Eigen::Matrix<bool, -1, 1, 0, -1, 1>&,
                      elsa::FiniteDifferences<std::complex<double>>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("activeDims"), py::arg("type"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::DataDescriptor&,
                      elsa::FiniteDifferences<std::complex<double>>::DiffType>(),
             py::arg("domainDescriptor"), py::arg("type"));

    py::class_<elsa::FourierTransform<std::complex<float>>,
               elsa::LinearOperator<std::complex<float>>>
        FourierTransformcf(m, "FourierTransformcf");
    FourierTransformcf
        .def(
            "set",
            (elsa::FourierTransform<std::complex<
                 float>> & (elsa::FourierTransform<std::complex<float>>::*) (const elsa::FourierTransform<std::complex<float>>&) )(
                &elsa::FourierTransform<std::complex<float>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, elsa::FFTNorm>(), py::arg("domainDescriptor"),
             py::arg("norm") = static_cast<elsa::FFTNorm>(2))
        .def(py::init<const elsa::FourierTransform<std::complex<float>>&>());

    m.attr("FourierTransform") = m.attr("FourierTransformcf");

    py::class_<elsa::FourierTransform<std::complex<double>>,
               elsa::LinearOperator<std::complex<double>>>
        FourierTransformcd(m, "FourierTransformcd");
    FourierTransformcd
        .def(
            "set",
            (elsa::FourierTransform<std::complex<
                 double>> & (elsa::FourierTransform<std::complex<double>>::*) (const elsa::FourierTransform<std::complex<double>>&) )(
                &elsa::FourierTransform<std::complex<double>>::operator=),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataDescriptor&, elsa::FFTNorm>(), py::arg("domainDescriptor"),
             py::arg("norm") = static_cast<elsa::FFTNorm>(2))
        .def(py::init<const elsa::FourierTransform<std::complex<double>>&>());

    py::class_<elsa::BlockLinearOperator<float>, elsa::LinearOperator<float>> BlockLinearOperatorf(
        m, "BlockLinearOperatorf");
    BlockLinearOperatorf
        .def("getIthOperator",
             (const elsa::LinearOperator<float>& (elsa::BlockLinearOperator<float>::*) (long)
                  const)(&elsa::BlockLinearOperator<float>::getIthOperator),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("numberOfOps", (long(elsa::BlockLinearOperator<float>::*)()
                                 const)(&elsa::BlockLinearOperator<float>::numberOfOps));

    elsa::BlockLinearOperatorHints<float>::addCustomMethods(BlockLinearOperatorf);

    m.attr("BlockLinearOperator") = m.attr("BlockLinearOperatorf");

    py::class_<elsa::BlockLinearOperator<double>, elsa::LinearOperator<double>>
        BlockLinearOperatord(m, "BlockLinearOperatord");
    BlockLinearOperatord
        .def("getIthOperator",
             (const elsa::LinearOperator<double>& (elsa::BlockLinearOperator<double>::*) (long)
                  const)(&elsa::BlockLinearOperator<double>::getIthOperator),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("numberOfOps", (long(elsa::BlockLinearOperator<double>::*)()
                                 const)(&elsa::BlockLinearOperator<double>::numberOfOps));

    elsa::BlockLinearOperatorHints<double>::addCustomMethods(BlockLinearOperatord);

    py::class_<elsa::BlockLinearOperator<std::complex<float>>,
               elsa::LinearOperator<std::complex<float>>>
        BlockLinearOperatorcf(m, "BlockLinearOperatorcf");
    BlockLinearOperatorcf
        .def("getIthOperator",
             (const elsa::LinearOperator<std::complex<float>>& (
                 elsa::BlockLinearOperator<std::complex<float>>::*) (long)
                  const)(&elsa::BlockLinearOperator<std::complex<float>>::getIthOperator),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("numberOfOps", (long(elsa::BlockLinearOperator<std::complex<float>>::*)() const)(
                                &elsa::BlockLinearOperator<std::complex<float>>::numberOfOps));

    elsa::BlockLinearOperatorHints<std::complex<float>>::addCustomMethods(BlockLinearOperatorcf);

    py::class_<elsa::BlockLinearOperator<std::complex<double>>,
               elsa::LinearOperator<std::complex<double>>>
        BlockLinearOperatorcd(m, "BlockLinearOperatorcd");
    BlockLinearOperatorcd
        .def("getIthOperator",
             (const elsa::LinearOperator<std::complex<double>>& (
                 elsa::BlockLinearOperator<std::complex<double>>::*) (long)
                  const)(&elsa::BlockLinearOperator<std::complex<double>>::getIthOperator),
             py::arg("index"), py::return_value_policy::reference_internal)
        .def("numberOfOps", (long(elsa::BlockLinearOperator<std::complex<double>>::*)() const)(
                                &elsa::BlockLinearOperator<std::complex<double>>::numberOfOps));

    elsa::BlockLinearOperatorHints<std::complex<double>>::addCustomMethods(BlockLinearOperatorcd);

    py::class_<elsa::Dictionary<float>, elsa::LinearOperator<float>> Dictionaryf(m, "Dictionaryf");
    Dictionaryf
        .def("getSupportedDictionary",
             (elsa::Dictionary<float>(elsa::Dictionary<float>::*)(
                 Eigen::Matrix<long, -1, 1, 0, -1, 1>)
                  const)(&elsa::Dictionary<float>::getSupportedDictionary),
             py::arg("support"), py::return_value_policy::move)
        .def("getAtom",
             (const elsa::DataContainer<float> (elsa::Dictionary<float>::*)(long)
                  const)(&elsa::Dictionary<float>::getAtom),
             py::arg("j"), py::return_value_policy::move)
        .def("getNumberOfAtoms",
             (long(elsa::Dictionary<float>::*)() const)(&elsa::Dictionary<float>::getNumberOfAtoms))
        .def(py::init<const elsa::DataContainer<float>&>(), py::arg("dictionary"))
        .def(py::init<const elsa::DataDescriptor&, long>(), py::arg("signalDescriptor"),
             py::arg("nAtoms"))
        .def("updateAtom",
             (void(elsa::Dictionary<float>::*)(long, const elsa::DataContainer<float>&))(
                 &elsa::Dictionary<float>::updateAtom),
             py::arg("j"), py::arg("atom"));

    m.attr("Dictionary") = m.attr("Dictionaryf");

    py::class_<elsa::Dictionary<double>, elsa::LinearOperator<double>> Dictionaryd(m,
                                                                                   "Dictionaryd");
    Dictionaryd
        .def("getSupportedDictionary",
             (elsa::Dictionary<double>(elsa::Dictionary<double>::*)(
                 Eigen::Matrix<long, -1, 1, 0, -1, 1>)
                  const)(&elsa::Dictionary<double>::getSupportedDictionary),
             py::arg("support"), py::return_value_policy::move)
        .def("getAtom",
             (const elsa::DataContainer<double> (elsa::Dictionary<double>::*)(long)
                  const)(&elsa::Dictionary<double>::getAtom),
             py::arg("j"), py::return_value_policy::move)
        .def("getNumberOfAtoms", (long(elsa::Dictionary<double>::*)()
                                      const)(&elsa::Dictionary<double>::getNumberOfAtoms))
        .def(py::init<const elsa::DataContainer<double>&>(), py::arg("dictionary"))
        .def(py::init<const elsa::DataDescriptor&, long>(), py::arg("signalDescriptor"),
             py::arg("nAtoms"))
        .def("updateAtom",
             (void(elsa::Dictionary<double>::*)(long, const elsa::DataContainer<double>&))(
                 &elsa::Dictionary<double>::updateAtom),
             py::arg("j"), py::arg("atom"));

    py::class_<elsa::ShearletTransform<float, float>, elsa::LinearOperator<float>>
        ShearletTransformfloatfloat(m, "ShearletTransformfloatfloat");
    ShearletTransformfloatfloat
        .def("isSpectraComputed", (bool(elsa::ShearletTransform<float, float>::*)() const)(
                                      &elsa::ShearletTransform<float, float>::isSpectraComputed))
        .def("getSpectra",
             (elsa::DataContainer<float>(elsa::ShearletTransform<float, float>::*)()
                  const)(&elsa::ShearletTransform<float, float>::getSpectra),
             py::return_value_policy::move)
        .def("sumByLastAxis",
             (elsa::DataContainer<std::complex<float>>(elsa::ShearletTransform<float, float>::*)(
                 elsa::DataContainer<std::complex<float>>)
                  const)(&elsa::ShearletTransform<float, float>::sumByLastAxis),
             py::arg("dc"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::ShearletTransform<
                 float,
                 float> & (elsa::ShearletTransform<float, float>::*) (const elsa::ShearletTransform<float, float>&) )(
                &elsa::ShearletTransform<float, float>::operator=),
            py::return_value_policy::reference_internal)
        .def("getHeight", (long(elsa::ShearletTransform<float, float>::*)()
                               const)(&elsa::ShearletTransform<float, float>::getHeight))
        .def("getNumOfLayers", (long(elsa::ShearletTransform<float, float>::*)()
                                    const)(&elsa::ShearletTransform<float, float>::getNumOfLayers))
        .def("getWidth", (long(elsa::ShearletTransform<float, float>::*)()
                              const)(&elsa::ShearletTransform<float, float>::getWidth))
        .def(py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>>(), py::arg("spatialDimensions"))
        .def(py::init<const elsa::ShearletTransform<float, float>&>())
        .def(py::init<long, long>(), py::arg("width"), py::arg("height"))
        .def(py::init<long, long, long>(), py::arg("width"), py::arg("height"),
             py::arg("numOfScales"))
        .def(py::init<long, long, long, std::optional<elsa::DataContainer<float>>>(),
             py::arg("width"), py::arg("height"), py::arg("numOfScales"), py::arg("spectra"))
        .def("computeSpectra", (void(elsa::ShearletTransform<float, float>::*)()
                                    const)(&elsa::ShearletTransform<float, float>::computeSpectra));

    m.attr("ShearletTransform") = m.attr("ShearletTransformfloatfloat");

    py::class_<elsa::ShearletTransform<std::complex<float>, float>,
               elsa::LinearOperator<std::complex<float>>>
        ShearletTransformcffloat(m, "ShearletTransformcffloat");
    ShearletTransformcffloat
        .def("isSpectraComputed",
             (bool(elsa::ShearletTransform<std::complex<float>, float>::*)()
                  const)(&elsa::ShearletTransform<std::complex<float>, float>::isSpectraComputed))
        .def("getSpectra",
             (elsa::DataContainer<float>(elsa::ShearletTransform<std::complex<float>, float>::*)()
                  const)(&elsa::ShearletTransform<std::complex<float>, float>::getSpectra),
             py::return_value_policy::move)
        .def("sumByLastAxis",
             (elsa::DataContainer<std::complex<float>>(
                 elsa::ShearletTransform<std::complex<float>, float>::*)(
                 elsa::DataContainer<std::complex<float>>)
                  const)(&elsa::ShearletTransform<std::complex<float>, float>::sumByLastAxis),
             py::arg("dc"), py::return_value_policy::move)
        .def(
            "set",
            (elsa::ShearletTransform<
                 std::complex<float>,
                 float> & (elsa::ShearletTransform<std::complex<float>, float>::*) (const elsa::ShearletTransform<std::complex<float>, float>&) )(
                &elsa::ShearletTransform<std::complex<float>, float>::operator=),
            py::return_value_policy::reference_internal)
        .def("getHeight", (long(elsa::ShearletTransform<std::complex<float>, float>::*)() const)(
                              &elsa::ShearletTransform<std::complex<float>, float>::getHeight))
        .def("getNumOfLayers",
             (long(elsa::ShearletTransform<std::complex<float>, float>::*)()
                  const)(&elsa::ShearletTransform<std::complex<float>, float>::getNumOfLayers))
        .def("getWidth", (long(elsa::ShearletTransform<std::complex<float>, float>::*)() const)(
                             &elsa::ShearletTransform<std::complex<float>, float>::getWidth))
        .def(py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>>(), py::arg("spatialDimensions"))
        .def(py::init<const elsa::ShearletTransform<std::complex<float>, float>&>())
        .def(py::init<long, long>(), py::arg("width"), py::arg("height"))
        .def(py::init<long, long, long>(), py::arg("width"), py::arg("height"),
             py::arg("numOfScales"))
        .def(py::init<long, long, long, std::optional<elsa::DataContainer<float>>>(),
             py::arg("width"), py::arg("height"), py::arg("numOfScales"), py::arg("spectra"))
        .def("computeSpectra",
             (void(elsa::ShearletTransform<std::complex<float>, float>::*)()
                  const)(&elsa::ShearletTransform<std::complex<float>, float>::computeSpectra));

    py::class_<elsa::ShearletTransform<double, double>, elsa::LinearOperator<double>>
        ShearletTransformdoubledouble(m, "ShearletTransformdoubledouble");
    ShearletTransformdoubledouble
        .def("isSpectraComputed", (bool(elsa::ShearletTransform<double, double>::*)() const)(
                                      &elsa::ShearletTransform<double, double>::isSpectraComputed))
        .def("sumByLastAxis",
             (elsa::DataContainer<std::complex<double>>(elsa::ShearletTransform<double, double>::*)(
                 elsa::DataContainer<std::complex<double>>)
                  const)(&elsa::ShearletTransform<double, double>::sumByLastAxis),
             py::arg("dc"), py::return_value_policy::move)
        .def("getSpectra",
             (elsa::DataContainer<double>(elsa::ShearletTransform<double, double>::*)()
                  const)(&elsa::ShearletTransform<double, double>::getSpectra),
             py::return_value_policy::move)
        .def(
            "set",
            (elsa::ShearletTransform<
                 double,
                 double> & (elsa::ShearletTransform<double, double>::*) (const elsa::ShearletTransform<double, double>&) )(
                &elsa::ShearletTransform<double, double>::operator=),
            py::return_value_policy::reference_internal)
        .def("getHeight", (long(elsa::ShearletTransform<double, double>::*)()
                               const)(&elsa::ShearletTransform<double, double>::getHeight))
        .def("getNumOfLayers", (long(elsa::ShearletTransform<double, double>::*)() const)(
                                   &elsa::ShearletTransform<double, double>::getNumOfLayers))
        .def("getWidth", (long(elsa::ShearletTransform<double, double>::*)()
                              const)(&elsa::ShearletTransform<double, double>::getWidth))
        .def(py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>>(), py::arg("spatialDimensions"))
        .def(py::init<const elsa::ShearletTransform<double, double>&>())
        .def(py::init<long, long>(), py::arg("width"), py::arg("height"))
        .def(py::init<long, long, long>(), py::arg("width"), py::arg("height"),
             py::arg("numOfScales"))
        .def(py::init<long, long, long, std::optional<elsa::DataContainer<double>>>(),
             py::arg("width"), py::arg("height"), py::arg("numOfScales"), py::arg("spectra"))
        .def("computeSpectra", (void(elsa::ShearletTransform<double, double>::*)() const)(
                                   &elsa::ShearletTransform<double, double>::computeSpectra));

    py::class_<elsa::ShearletTransform<std::complex<double>, double>,
               elsa::LinearOperator<std::complex<double>>>
        ShearletTransformcddouble(m, "ShearletTransformcddouble");
    ShearletTransformcddouble
        .def("isSpectraComputed",
             (bool(elsa::ShearletTransform<std::complex<double>, double>::*)()
                  const)(&elsa::ShearletTransform<std::complex<double>, double>::isSpectraComputed))
        .def("sumByLastAxis",
             (elsa::DataContainer<std::complex<double>>(
                 elsa::ShearletTransform<std::complex<double>, double>::*)(
                 elsa::DataContainer<std::complex<double>>)
                  const)(&elsa::ShearletTransform<std::complex<double>, double>::sumByLastAxis),
             py::arg("dc"), py::return_value_policy::move)
        .def(
            "getSpectra",
            (elsa::DataContainer<double>(elsa::ShearletTransform<std::complex<double>, double>::*)()
                 const)(&elsa::ShearletTransform<std::complex<double>, double>::getSpectra),
            py::return_value_policy::move)
        .def(
            "set",
            (elsa::ShearletTransform<
                 std::complex<double>,
                 double> & (elsa::ShearletTransform<std::complex<double>, double>::*) (const elsa::ShearletTransform<std::complex<double>, double>&) )(
                &elsa::ShearletTransform<std::complex<double>, double>::operator=),
            py::return_value_policy::reference_internal)
        .def("getHeight", (long(elsa::ShearletTransform<std::complex<double>, double>::*)() const)(
                              &elsa::ShearletTransform<std::complex<double>, double>::getHeight))
        .def("getNumOfLayers",
             (long(elsa::ShearletTransform<std::complex<double>, double>::*)()
                  const)(&elsa::ShearletTransform<std::complex<double>, double>::getNumOfLayers))
        .def("getWidth", (long(elsa::ShearletTransform<std::complex<double>, double>::*)() const)(
                             &elsa::ShearletTransform<std::complex<double>, double>::getWidth))
        .def(py::init<Eigen::Matrix<long, -1, 1, 0, -1, 1>>(), py::arg("spatialDimensions"))
        .def(py::init<const elsa::ShearletTransform<std::complex<double>, double>&>())
        .def(py::init<long, long>(), py::arg("width"), py::arg("height"))
        .def(py::init<long, long, long>(), py::arg("width"), py::arg("height"),
             py::arg("numOfScales"))
        .def(py::init<long, long, long, std::optional<elsa::DataContainer<double>>>(),
             py::arg("width"), py::arg("height"), py::arg("numOfScales"), py::arg("spectra"))
        .def("computeSpectra",
             (void(elsa::ShearletTransform<std::complex<double>, double>::*)()
                  const)(&elsa::ShearletTransform<std::complex<double>, double>::computeSpectra));

    elsa::OperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_operators, m)
{
    add_definitions_pyelsa_operators(m);
}
