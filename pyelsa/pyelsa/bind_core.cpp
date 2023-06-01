#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "Cloneable.h"
#include "Complex.h"
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

#include "bind_common.h"
#include "elsaDefines.h"
#include "hints/core_hints.cpp"

namespace py = pybind11;

namespace detail
{
    template <class data_t>
    void add_clonable_linear_operator(py::module& m, const char* name)
    {
        using CloneableLOp = elsa::Cloneable<elsa::LinearOperator<data_t>>;
        using ConstLOpRef = const elsa::LinearOperator<data_t>&;

        py::class_<CloneableLOp> op(m, name);
        op.def("__ne__", py::overload_cast<ConstLOpRef>(&CloneableLOp::operator!=, py::const_),
               py::arg("other"));
        op.def("__eq__", py::overload_cast<ConstLOpRef>(&CloneableLOp::operator==, py::const_),
               py::arg("other"));
        op.def("clone", py::overload_cast<>(&CloneableLOp ::clone, py::const_));
    }

    template <class data_t>
    void add_linear_operator(py::module& m, const char* name)
    {
        using LOp = elsa::LinearOperator<data_t>;
        using DcRef = elsa::DataContainer<data_t>&;
        using ConstDcRef = const elsa::DataContainer<data_t>&;

        auto return_move = py::return_value_policy::move;
        auto ref_internal = py::return_value_policy::reference_internal;

        py::class_<LOp, elsa::Cloneable<LOp>> op(m, name);
        op.def("apply", py::overload_cast<ConstDcRef>(&LOp::apply, py::const_), py::arg("x"),
               return_move);
        op.def("apply", py::overload_cast<ConstDcRef, DcRef>(&LOp::apply, py::const_), py::arg("x"),
               py::arg("Ax"));
        op.def("applyAdjoint", py::overload_cast<ConstDcRef>(&LOp::applyAdjoint, py::const_),
               py::arg("y"), return_move);
        op.def("applyAdjoint", py::overload_cast<ConstDcRef, DcRef>(&LOp::applyAdjoint, py::const_),
               py::arg("y"), py::arg("Aty"));
        op.def("set", py::overload_cast<const LOp&>(&LOp::operator=), py::arg("other"),
               ref_internal);
        op.def("getDomainDescriptor", py::overload_cast<>(&LOp::getDomainDescriptor, py::const_),
               ref_internal);
        op.def("getRangeDescriptor", py::overload_cast<>(&LOp::getRangeDescriptor, py::const_),
               ref_internal);
        op.def(py::init<const elsa::DataDescriptor&, const elsa::DataDescriptor&>(),
               py::arg("domainDescriptor"), py::arg("rangeDescriptor"));
        op.def(py::init<const elsa::LinearOperator<data_t>&>(), py::arg("other"));

        elsa::LinearOperatorHints<data_t>::addCustomMethods(op);
    }

    template <class data_t>
    void add_data_container(py::module& m, const char* name)
    {
        using Dc = elsa::DataContainer<data_t>;
        using ConstDcRef = const Dc&;
        using IndexVec = Eigen::Matrix<long, Eigen::Dynamic, 1>;

        py::class_<Dc> dc(m, name, py::buffer_protocol());

        const auto ref_internal = py::return_value_policy::reference_internal;
        const auto move = py::return_value_policy::move;

        dc.def("__setitem__", [](Dc& self, const Dc& other) { self = other; });
        dc.def("__setitem__", [](Dc& self, data_t scalar) { self = scalar; });

        // Element Access
        dc.def("__setitem__", [](Dc& self, elsa::index_t idx, data_t val) { self[idx] = val; });
        dc.def("__getitem__", [](Dc& self, elsa::index_t idx) { return self[idx]; });

        // Equality comparison
        dc.def(py::self == py::self);
        dc.def(py::self != py::self);

        // Inplace vector operations
        dc.def("set", py::overload_cast<ConstDcRef>(&Dc::operator=), py::arg("other"),
               ref_internal);
        dc.def(py::self += py::self);
        dc.def(py::self -= py::self);
        dc.def(py::self *= py::self);
        dc.def(py::self /= py::self);

        // Inplace scalar operations
        dc.def("set", py::overload_cast<data_t>(&Dc::operator=), py::arg("scalar"), ref_internal);
        dc.def(py::self += data_t());
        dc.def(py::self -= data_t());
        dc.def(py::self *= data_t());
        dc.def(py::self /= data_t());

        // Unary operations
        dc.def(+py::self);
        dc.def(-py::self);

        // Binary operations
        dc.def(py::self + py::self);
        dc.def(py::self - py::self);
        dc.def(py::self * py::self);
        dc.def(py::self / py::self);

        dc.def(py::self + data_t());
        dc.def(py::self - data_t());
        dc.def(py::self * data_t());
        dc.def(py::self / data_t());

        dc.def(data_t() + py::self);
        dc.def(data_t() - py::self);
        dc.def(data_t() * py::self);
        dc.def(data_t() / py::self);

        // Block operations
        dc.def("viewAs", py::overload_cast<const elsa::DataDescriptor&>(&Dc::viewAs),
               py::arg("descriptor"), move);
        dc.def("viewAs", py::overload_cast<const elsa::DataDescriptor&>(&Dc::viewAs, py::const_),
               py::arg("descriptor"), move);
        dc.def("getBlock", py::overload_cast<long>(&Dc::getBlock), py::arg("i"), move);
        dc.def("getBlock", py::overload_cast<long>(&Dc::getBlock, py::const_), py::arg("i"), move);
        dc.def("slice", py::overload_cast<long>(&Dc::slice), py::arg("i"), move);
        dc.def("slice", py::overload_cast<long>(&Dc::slice, py::const_), py::arg("i"), move);

        dc.def("getDataDescriptor", py::overload_cast<>(&Dc::getDataDescriptor, py::const_),
               ref_internal);

        dc.def("at", py::overload_cast<const IndexVec&>(&Dc::at, py::const_),
               py::arg("coordinate"));
        dc.def("getSize", py::overload_cast<>(&Dc::getSize, py::const_));

        dc.def("dot", py::overload_cast<ConstDcRef>(&Dc::dot, py::const_), py::arg("other"));
        dc.def("l1Norm", py::overload_cast<>(&Dc::l1Norm, py::const_));
        dc.def("l2Norm", py::overload_cast<>(&Dc::l2Norm, py::const_));
        dc.def("lInfNorm", py::overload_cast<>(&Dc::lInfNorm, py::const_));
        dc.def("maxElement", py::overload_cast<>(&Dc::maxElement, py::const_));
        dc.def("minElement", py::overload_cast<>(&Dc::minElement, py::const_));
        dc.def("squaredL2Norm", py::overload_cast<>(&Dc::squaredL2Norm, py::const_));
        dc.def("sum", py::overload_cast<>(&Dc::sum, py::const_));
        dc.def("l0PseudoNorm", py::overload_cast<>(&Dc::l0PseudoNorm, py::const_));

        dc.def("fft", py::overload_cast<elsa::FFTNorm>(&Dc::fft), py::arg("norm"));
        dc.def("ifft", py::overload_cast<elsa::FFTNorm>(&Dc::ifft), py::arg("norm"));

        using ostream = std::basic_ostream<char, std::char_traits<char>>;
        dc.def("format", py::overload_cast<ostream&, elsa::format_config>(&Dc::format, py::const_),
               py::arg("os"), py::arg("cfg") = elsa::format_config{});

        using Vector = Eigen::Matrix<data_t, Eigen::Dynamic, 1>;
        using ConstDescRef = const elsa::DataDescriptor&;

        dc.def(py::init<ConstDcRef>(), py::arg("other"));
        dc.def(py::init<ConstDescRef, const Vector&>(), py::arg("dataDescriptor"), py::arg("data"));
        dc.def(py::init<ConstDescRef>(), py::arg("dataDescriptor"));

        elsa::DataContainerHints<data_t>::addCustomMethods(dc);

        elsa::DataContainerHints<data_t>::exposeBufferInfo(dc);
    }
} // namespace detail

void add_linear_operator(py::module& m)
{
    detail::add_clonable_linear_operator<float>(m, "CloneableLinearOperatorf");
    detail::add_linear_operator<float>(m, "LinearOperatorf");
    m.attr("LinearOperator") = m.attr("LinearOperatorf");

    detail::add_clonable_linear_operator<thrust::complex<float>>(m, "CloneableLinearOperatorcf");
    detail::add_linear_operator<thrust::complex<float>>(m, "LinearOperatorcf");

    detail::add_clonable_linear_operator<double>(m, "CloneableLinearOperatord");
    detail::add_linear_operator<double>(m, "LinearOperatord");

    detail::add_clonable_linear_operator<thrust::complex<double>>(m, "CloneableLinearOperatorcd");
    detail::add_linear_operator<thrust::complex<double>>(m, "LinearOperatorcd");
}

namespace detail
{
    template <class Fn>
    void add_fn(py::module& m, std::string name, Fn fn)
    {
        m.def(name.c_str(), fn);
    }
} // namespace detail

void add_datacontainer_free_functions(py::module& m)
{
    detail::add_fn(m, "clip", elsa::clip<float>);
    detail::add_fn(m, "clip", elsa::clip<double>);

#define ELSA_ADD_FREE_FUNCTION_ALL_TYPES(name)                  \
    detail::add_fn(m, #name, elsa::name<elsa::index_t>);        \
    detail::add_fn(m, #name, elsa::name<float>);                \
    detail::add_fn(m, #name, elsa::name<double>);               \
    detail::add_fn(m, #name, elsa::name<elsa::complex<float>>); \
    detail::add_fn(m, #name, elsa::name<elsa::complex<double>>);

    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(minimum)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(maximum)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(real)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(imag)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(cwiseAbs)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(square)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(exp)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(log)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(empty)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(emptylike)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(zeros)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(zeroslike)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(ones)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(oneslike)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(empty)
    ELSA_ADD_FREE_FUNCTION_ALL_TYPES(emptylike)

    // Why doesn't this work, I don't know...but whatever, I DOOOO NOT CAAAARE
    /* m.def("sqrt", elsa::sqrt<elsa::index_t>); */
    /* m.def("sqrt", elsa::sqrt<float>); */
    /* m.def("sqrt", elsa::sqrt<double>); */
    /* m.def("sqrt", elsa::sqrt<elsa::complex<float>>); */
    /* m.def("sqrt", elsa::sqrt<elsa::complex<double>>); */

#undef ELSA_ADD_FREE_FUNCTION

#define ELSA_ADD_FREE_FUNCTIONS(name)                                      \
    m.def(#name, elsa::name<elsa::index_t, elsa::index_t>);                \
    m.def(#name, elsa::name<elsa::index_t, float>);                        \
    m.def(#name, elsa::name<elsa::index_t, double>);                       \
    m.def(#name, elsa::name<elsa::index_t, elsa::complex<float>>);         \
    m.def(#name, elsa::name<elsa::index_t, elsa::complex<double>>);        \
    m.def(#name, elsa::name<float, float>);                                \
    m.def(#name, elsa::name<float, double>);                               \
    m.def(#name, elsa::name<float, elsa::complex<float>>);                 \
    m.def(#name, elsa::name<float, elsa::complex<double>>);                \
    m.def(#name, elsa::name<double, double>);                              \
    m.def(#name, elsa::name<double, elsa::complex<float>>);                \
    m.def(#name, elsa::name<double, elsa::complex<double>>);               \
    m.def(#name, elsa::name<elsa::complex<float>, elsa::complex<float>>);  \
    m.def(#name, elsa::name<elsa::complex<float>, elsa::complex<double>>); \
    m.def(#name, elsa::name<elsa::complex<double>, elsa::complex<double>>);

    ELSA_ADD_FREE_FUNCTIONS(cwiseMin)
    ELSA_ADD_FREE_FUNCTIONS(cwiseMax)

#undef ELSA_ADD_FREE_FUNCTIONS
}

void add_data_container(py::module& m)
{
    detail::add_data_container<float>(m, "DataContainerf");
    m.attr("DataContainer") = m.attr("DataContainerf");

    detail::add_data_container<double>(m, "DataContainerd");
    detail::add_data_container<thrust::complex<float>>(m, "DataContainercf");
    detail::add_data_container<thrust::complex<double>>(m, "DataContainercd");
    detail::add_data_container<long>(m, "DataContainerl");

    add_datacontainer_free_functions(m);
}

void add_definitions_pyelsa_core(py::module& m)
{
    py::enum_<elsa::FFTNorm>(m, "FFTNorm")
        .value("BACKWARD", elsa::FFTNorm::BACKWARD)
        .value("FORWARD", elsa::FFTNorm::FORWARD)
        .value("ORTHO", elsa::FFTNorm::ORTHO);

    using DD = elsa::DataDescriptor;
    py::class_<elsa::Cloneable<DD>> CloneableDataDescriptor(m, "CloneableDataDescriptor");
    CloneableDataDescriptor
        .def("__ne__", py::overload_cast<const DD&>(&elsa::Cloneable<DD>::operator!=, py::const_),
             py::arg("other"))
        .def("__eq__", py::overload_cast<const DD&>(&elsa::Cloneable<DD>::operator==, py::const_),
             py::arg("other"))
        .def("clone", py::overload_cast<>(&elsa::Cloneable<DD>::clone, py::const_));

    using IndexVec = elsa::IndexVector_t;
    const auto move = py::return_value_policy::move;
    py::class_<DD, elsa::Cloneable<DD>> dd(m, "DataDescriptor");
    dd.def("getLocationOfOrigin", py::overload_cast<>(&DD::getLocationOfOrigin, py::const_), move);
    dd.def("getSpacingPerDimension", py::overload_cast<>(&DD::getSpacingPerDimension, py::const_),
           move);
    dd.def("getCoordinateFromIndex",
           py::overload_cast<long>(&DD::getCoordinateFromIndex, py::const_), py::arg("index"),
           move);
    dd.def("getNumberOfCoefficientsPerDimension",
           py::overload_cast<>(&DD::getNumberOfCoefficientsPerDimension, py::const_), move);
    dd.def("getIndexFromCoordinate",
           py::overload_cast<const IndexVec&>(&DD::getIndexFromCoordinate, py::const_),
           py::arg("coordinate"));
    dd.def("getNumberOfCoefficients",
           py::overload_cast<>(&DD::getNumberOfCoefficients, py::const_));
    dd.def("getNumberOfDimensions", py::overload_cast<>(&DD::getNumberOfDimensions, py::const_));
    dd.def("element", &DD::element<elsa::index_t>);
    dd.def("element", &DD::element<float>);
    dd.def("element", &DD::element<double>);
    dd.def("element", &DD::element<elsa::complex<float>>);
    dd.def("element", &DD::element<elsa::complex<double>>);

    py::class_<elsa::format_config> format_config(m, "format_config");
    format_config
        .def("set", py::overload_cast<const elsa::format_config&>(&elsa::format_config::operator=),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::format_config&>());

    add_data_container(m);
    add_linear_operator(m);

    py::class_<elsa::VolumeDescriptor, DD> VolumeDescriptor(m, "VolumeDescriptor");
    VolumeDescriptor.def(py::init<IndexVec>(), py::arg("numberOfCoefficientsPerDimension"))
        .def(py::init<IndexVec, elsa::RealVector_t>(), py::arg("numberOfCoefficientsPerDimension"),
             py::arg("spacingPerDimension"))
        .def(py::init<std::initializer_list<long>>(), py::arg("numberOfCoefficientsPerDimension"))
        .def(py::init<std::initializer_list<long>, std::initializer_list<float>>(),
             py::arg("numberOfCoefficientsPerDimension"), py::arg("spacingPerDimension"))
        .def(py::init<const elsa::VolumeDescriptor&>());

    using BlockDesc = elsa::BlockDescriptor;
    py::class_<BlockDesc, DD> blockDesc(m, "BlockDescriptor");
    blockDesc.def("getDescriptorOfBlock",
                  py::overload_cast<long>(&BlockDesc::getDescriptorOfBlock, py::const_),
                  py::arg("i"), py::return_value_policy::reference_internal);
    blockDesc.def("getOffsetOfBlock",
                  py::overload_cast<long>(&BlockDesc::getOffsetOfBlock, py::const_), py::arg("i"));
    blockDesc.def("getNumberOfBlocks",
                  py::overload_cast<>(&BlockDesc::getNumberOfBlocks, py::const_));

    using IdBlockDesc = elsa::IdenticalBlocksDescriptor;
    py::class_<IdBlockDesc, BlockDesc> idBlockDesc(m, "IdenticalBlocksDescriptor");
    idBlockDesc.def("getDescriptorOfBlock",
                    py::overload_cast<long>(&IdBlockDesc::getDescriptorOfBlock, py::const_),
                    py::arg("i"), py::return_value_policy::reference_internal);
    idBlockDesc.def("getOffsetOfBlock",
                    py::overload_cast<long>(&IdBlockDesc::getOffsetOfBlock, py::const_),
                    py::arg("i"));
    idBlockDesc.def(
        "getNumberOfBlocks",
        py::overload_cast<>(&elsa::IdenticalBlocksDescriptor::getNumberOfBlocks, py::const_));
    idBlockDesc.def(py::init<long, const elsa::DataDescriptor&>(), py::arg("numberOfBlocks"),
                    py::arg("dataDescriptor"));

    using PartDesc = elsa::PartitionDescriptor;
    py::class_<PartDesc, BlockDesc> partDesc(m, "PartitionDescriptor");
    partDesc
        .def("getDescriptorOfBlock",
             py::overload_cast<long>(&PartDesc::getDescriptorOfBlock, py::const_), py::arg("i"),
             py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock", py::overload_cast<long>(&PartDesc::getOffsetOfBlock, py::const_),
             py::arg("i"))
        .def("getNumberOfBlocks", py::overload_cast<>(&PartDesc::getNumberOfBlocks, py::const_))
        .def(py::init<const DD&, elsa::IndexVector_t>(), py::arg("dataDescriptor"),
             py::arg("slicesInBlock"))
        .def(py::init<const DD&, long>(), py::arg("dataDescriptor"), py::arg("numberOfBlocks"));

    using RandBlockDesc = elsa::RandomBlocksDescriptor;
    py::class_<RandBlockDesc, BlockDesc> randBlockDesc(m, "RandomBlocksDescriptor");
    randBlockDesc
        .def("getDescriptorOfBlock",
             py::overload_cast<long>(&RandBlockDesc::getDescriptorOfBlock, py::const_),
             py::arg("i"), py::return_value_policy::reference_internal)
        .def("getOffsetOfBlock",
             py::overload_cast<long>(&RandBlockDesc::getOffsetOfBlock, py::const_), py::arg("i"))
        .def("getNumberOfBlocks",
             py::overload_cast<>(&RandBlockDesc::getNumberOfBlocks, py::const_));

    elsa::RandomBlocksDescriptorHints::addCustomMethods(randBlockDesc);

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
