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
#include "AXDTOperator.h"
#include "XGIDetectorDescriptor.h"
#include "ZeroOperator.h"

#include "bind_common.h"

#include "hints/operators_hints.cpp"

namespace py = pybind11;

namespace detail
{
    template <class data_t>
    void add_finite_diff_difftype(py::module& m, const char* name)
    {
        using Op = elsa::FiniteDifferences<data_t>;
        py::enum_<typename Op::DiffType> e(m, name);

        e.value("BACKWARD", Op::DiffType::BACKWARD);
        e.value("CENTRAL", Op::DiffType::CENTRAL);
        e.value("FORWARD", Op::DiffType::FORWARD);
    }
} // namespace detail

void add_finite_difference_difftype(py::module& m)
{
    detail::add_finite_diff_difftype<float>(m, "FiniteDifferencesfDiffType");
    detail::add_finite_diff_difftype<double>(m, "FiniteDifferencesfDiffTyped");
    detail::add_finite_diff_difftype<thrust::complex<float>>(m, "FiniteDifferencesfDiffTypecf");
    detail::add_finite_diff_difftype<thrust::complex<double>>(m, "FiniteDifferencesfDiffTypecd");
}

namespace detail
{
    template <class data_t>
    void add_identity_op(py::module& m, const char* name)
    {
        py::class_<elsa::Identity<data_t>, elsa::LinearOperator<data_t>> op(m, name);
        op.def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));
        op.def("set",
               py::overload_cast<const elsa::Identity<data_t>&>(&elsa::Identity<data_t>::operator=),
               py::return_value_policy::reference_internal);
    }
} // namespace detail

void add_identity(py::module& m)
{
    detail::add_identity_op<float>(m, "Identityf");
    detail::add_identity_op<double>(m, "Identityd");
    detail::add_identity_op<thrust::complex<float>>(m, "Identitycf");
    detail::add_identity_op<thrust::complex<double>>(m, "Identitycd");

    m.attr("Identity") = m.attr("Identityf");
}

namespace detail
{
    template <class data_t>
    void add_scaling_op(py::module& m, const char* name)
    {
        using Op = elsa::Scaling<data_t>;
        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);
        op.def("isIsotropic", py::overload_cast<>(&Op::isIsotropic, py::const_));
        op.def("getScaleFactors", py::overload_cast<>(&Op::getScaleFactors, py::const_),
               py::return_value_policy::reference_internal);
        op.def("getScaleFactor", py::overload_cast<>(&Op::getScaleFactor, py::const_));
        op.def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<data_t>&>(),
               py::arg("descriptor"), py::arg("scaleFactors"));
        op.def(py::init<const elsa::DataDescriptor&, data_t>(), py::arg("descriptor"),
               py::arg("scaleFactor"));
    }
} // namespace detail

void add_scaling(py::module& m)
{
    detail::add_scaling_op<float>(m, "Scalingf");
    detail::add_scaling_op<double>(m, "Scalingd");
    detail::add_scaling_op<thrust::complex<float>>(m, "Scalingcf");
    detail::add_scaling_op<thrust::complex<double>>(m, "Scalingcd");

    m.attr("Scaling") = m.attr("Scalingf");
}

namespace detail
{
    template <class data_t>
    void add_finite_diff_op(py::module& m, const char* name)
    {
        using Op = elsa::FiniteDifferences<data_t>;
        using BoolVector = Eigen::Matrix<bool, Eigen::Dynamic, 1>;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);
        op.def("set", py::overload_cast<const Op&>(&Op::operator=),
               py::return_value_policy::reference_internal);
        op.def(py::init<const elsa::DataDescriptor&, const BoolVector&>(),
               py::arg("domainDescriptor"), py::arg("activeDims"));
        op.def(py::init<const elsa::DataDescriptor&, const BoolVector&, typename Op::DiffType>(),
               py::arg("domainDescriptor"), py::arg("activeDims"), py::arg("type"));
        op.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"));
        op.def(py::init<const elsa::DataDescriptor&, typename Op::DiffType>(),
               py::arg("domainDescriptor"), py::arg("type"));
    }
} // namespace detail

void add_finite_difference(py::module& m)
{
    detail::add_finite_diff_op<float>(m, "FiniteDifferencesf");
    detail::add_finite_diff_op<double>(m, "FiniteDifferencesd");
    detail::add_finite_diff_op<thrust::complex<float>>(m, "FiniteDifferencescf");
    detail::add_finite_diff_op<thrust::complex<double>>(m, "FiniteDifferencescd");

    m.attr("FiniteDifferences") = m.attr("FiniteDifferencesf");
}

namespace detail
{
    template <class data_t>
    void add_fourier_op(py::module& m, const char* name)
    {
        using Op = elsa::FourierTransform<data_t>;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);
        op.def("set", py::overload_cast<const Op&>(&Op::operator=),
               py::return_value_policy::reference_internal);
        op.def(py::init<const elsa::DataDescriptor&, elsa::FFTNorm>(), py::arg("domainDescriptor"),
               py::arg("norm") = static_cast<elsa::FFTNorm>(2));
        op.def(py::init<const elsa::FourierTransform<data_t>&>());
    }
} // namespace detail

void add_fourier_transform(py::module& m)
{
    detail::add_fourier_op<thrust::complex<float>>(m, "FourierTransformcf");
    detail::add_fourier_op<thrust::complex<double>>(m, "FourierTransformcd");

    m.attr("FourierTransform") = m.attr("FourierTransformcf");
}

namespace detail
{
    template <class data_t>
    void add_block_op(py::module& m, const char* name)
    {
        using Op = elsa::BlockLinearOperator<data_t>;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);
        op.def("getIthOperator", py::overload_cast<long>(&Op::getIthOperator, py::const_),
               py::arg("index"), py::return_value_policy::reference_internal);
        op.def("numberOfOps", py::overload_cast<>(&Op::numberOfOps, py::const_));

        py::enum_<typename elsa::BlockLinearOperator<data_t>::BlockType> opEnum(op, "BlockType");
        opEnum.value("COL", elsa::BlockLinearOperator<data_t>::BlockType::COL);
        opEnum.value("ROW", elsa::BlockLinearOperator<data_t>::BlockType::ROW);
        opEnum.export_values();

        elsa::BlockLinearOperatorHints<data_t>::addCustomMethods(op);
    }
} // namespace detail

void add_block_op(py::module& m)
{
    detail::add_block_op<float>(m, "BlockLinearOperatorf");
    detail::add_block_op<double>(m, "BlockLinearOperatord");
    detail::add_block_op<thrust::complex<float>>(m, "BlockLinearOperatorcf");
    detail::add_block_op<thrust::complex<double>>(m, "BlockLinearOperatorcd");

    m.attr("BlockLinearOperator") = m.attr("BlockLinearOperatorf");
}

namespace detail
{
    template <class data_t>
    void add_dictionary_op(py::module& m, const char* name)
    {
        using Op = elsa::Dictionary<data_t>;
        using IndexVector_t = elsa::IndexVector_t;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);
        op.def("getSupportedDictionary",
               py::overload_cast<IndexVector_t>(&Op::getSupportedDictionary, py::const_),
               py::arg("support"), py::return_value_policy::move);
        op.def("getAtom", py::overload_cast<long>(&Op::getAtom, py::const_), py::arg("j"),
               py::return_value_policy::move);
        op.def("getNumberOfAtoms", py::overload_cast<>(&Op::getNumberOfAtoms, py::const_));
        op.def(py::init<const elsa::DataContainer<data_t>&>(), py::arg("dictionary"));
        op.def(py::init<const elsa::DataDescriptor&, long>(), py::arg("signalDescriptor"),
               py::arg("nAtoms"));
        op.def("updateAtom",
               py::overload_cast<long, const elsa::DataContainer<data_t>&>(&Op::updateAtom),
               py::arg("j"), py::arg("atom"));
    }
} // namespace detail

void add_dictionary(py::module& m)
{
    detail::add_dictionary_op<float>(m, "Dictionaryf");
    detail::add_dictionary_op<double>(m, "Dictionaryd");

    m.attr("Dictionary") = m.attr("Dictionaryf");
}

namespace detail
{
    template <class data_x_t, class data_y_t>
    void add_shearlet_op(py::module& m, const char* name)
    {
        using Op = elsa::ShearletTransform<data_x_t, data_y_t>;
        using IndexVector_t = elsa::IndexVector_t;

        py::class_<Op, elsa::LinearOperator<data_x_t>> op(m, name);

        op.def("isSpectraComputed", py::overload_cast<>(&Op::isSpectraComputed, py::const_));
        op.def("getSpectra", py::overload_cast<>(&Op::getSpectra, py::const_),
               py::return_value_policy::move);
        op.def("sumByLastAxis",
               py::overload_cast<elsa::DataContainer<thrust::complex<data_y_t>>>(&Op::sumByLastAxis,
                                                                                 py::const_),
               py::arg("dc"), py::return_value_policy::move);
        op.def("getHeight", py::overload_cast<>(&Op::getHeight, py::const_));
        op.def("getNumOfLayers", py::overload_cast<>(&Op::getNumOfLayers, py::const_));
        op.def("getWidth", py::overload_cast<>(&Op::getWidth, py::const_));
        op.def("computeSpectra", py::overload_cast<>(&Op::computeSpectra, py::const_));
        op.def(py::init<IndexVector_t>(), py::arg("spatialDimensions"));
        op.def(py::init<const Op&>());
        op.def(py::init<long, long>(), py::arg("width"), py::arg("height"));
        op.def(py::init<long, long, long>(), py::arg("width"), py::arg("height"),
               py::arg("numOfScales"));
        op.def(py::init<long, long, long, std::optional<elsa::DataContainer<data_y_t>>>(),
               py::arg("width"), py::arg("height"), py::arg("numOfScales"), py::arg("spectra"));
        op.def("set", py::overload_cast<const Op&>(&Op::operator=),
               py::return_value_policy::reference_internal);
    }
} // namespace detail

void add_shearlet_operator(py::module& m)
{
    detail::add_shearlet_op<float, float>(m, "ShearletTransformff");
    detail::add_shearlet_op<thrust::complex<float>, float>(m, "ShearletTransformcff");
    detail::add_shearlet_op<double, double>(m, "ShearletTransformdd");
    detail::add_shearlet_op<thrust::complex<double>, double>(m, "ShearletTransformcdd");

    m.attr("ShearletTransform") = m.attr("ShearletTransformff");
}

namespace detail
{
    template <typename data_t>
    void add_AXDT_op(py::module& m, const char* name)
    {
        using Op = elsa::AXDTOperator<data_t>;
        using DirVecList = typename Op::DirVecList;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);

        py::enum_<typename Op::Symmetry>(op, "Symmetry")
            .value("Even", Op::Symmetry::even)
            .value("Odd", Op::Symmetry::odd)
            .value("Regular", Op::Symmetry::regular)
            .export_values();

        op.def(
            py::init<const elsa::VolumeDescriptor&, const elsa::XGIDetectorDescriptor&,
                     const elsa::LinearOperator<data_t>&, const DirVecList&,
                     const elsa::Vector_t<data_t>&, const typename Op::Symmetry&, elsa::index_t>(),
            py::arg("domainDescriptor"), py::arg("rangeDescriptor"), py::arg("projector"),
            py::arg("sphericalFuncDirs"), py::arg("sphericalFuncWeights"),
            py::arg("sphericalHarmonicsSymmetry"), py::arg("sphericalHarmonicsMaxDegree"));
    }
} // namespace detail

void add_AXDT_operator(py::module& m)
{
    detail::add_AXDT_op<float>(m, "AXDTOperatorf");
    detail::add_AXDT_op<double>(m, "AXDTOperatord");

    m.attr("AXDTOperator") = m.attr("AXDTOperatorf");
}

namespace detail
{
    template <typename data_t>
    void add_Zero_op(py::module& m, const char* name)
    {
        using Op = elsa::ZeroOperator<data_t>;

        py::class_<Op, elsa::LinearOperator<data_t>> op(m, name);

        op.def(
            py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
            py::arg("domainDescriptor"), py::arg("rangeDescriptor"));
    }
} // namespace detail

void add_Zero_operator(py::module& m)
{
    detail::add_Zero_op<float>(m, "ZeroOperatorf");
    detail::add_Zero_op<double>(m, "ZeroOperatord");

    m.attr("ZeroOperator") = m.attr("ZeroOperatorf");
}

void add_definitions_pyelsa_operators(py::module& m)
{
    add_finite_difference_difftype(m);
    add_identity(m);
    add_scaling(m);
    add_finite_difference(m);
    add_fourier_transform(m);
    add_block_op(m);
    add_shearlet_operator(m);
    add_AXDT_operator(m);
    add_Zero_operator(m);

    elsa::OperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_operators, m)
{
    add_definitions_pyelsa_operators(m);
}
