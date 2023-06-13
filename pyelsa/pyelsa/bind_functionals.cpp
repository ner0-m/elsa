#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include "Complex.h"
#include "Constraint.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "EmissionLogLikelihood.h"
#include "Functional.h"
#include "Huber.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "L2Squared.h"
#include "L2Reg.h"
#include "LeastSquares.h"
#include "LInfNorm.h"
#include "LinearOperator.h"
#include "LinearResidual.h"
#include "PseudoHuber.h"
#include "Quadric.h"
#include "SeparableSum.h"
#include "TransmissionLogLikelihood.h"
#include "WeightedL1Norm.h"
<<<<<<< HEAD
#include "WeightedL2NormPow2.h"
#include "AXDTStatRecon.h"
=======
#include "WeightedLeastSquares.h"
#include "WeightedL2Squared.h"
>>>>>>> master

#include "hints/functionals_hints.cpp"

#include "spdlog/fmt/fmt.h"

#include <string_view>

#include "bind_common.h"

namespace py = pybind11;

namespace detail
{
    template <class data_t>
    void add_linear_residual(py::module& m, const char* name)
    {
        using LinRes = elsa::LinearResidual<data_t>;
        using DC = elsa::DataContainer<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<LinRes> linres(m, name);
        linres.def("hasDataVector", py::overload_cast<>(&LinRes::hasDataVector, py::const_));
        linres.def("hasOperator", py::overload_cast<>(&LinRes::hasOperator, py::const_));
        linres.def("getDataVector", py::overload_cast<>(&LinRes::getDataVector, py::const_),
                   py::return_value_policy::reference_internal);
        linres.def("getOperator", py::overload_cast<>(&LinRes::getOperator, py::const_),
                   py::return_value_policy::reference_internal);
        linres.def(py::init<const DC&>(), py::arg("b"));
        linres.def(py::init<const DC&>(), py::arg("descriptor"));
        linres.def(py::init<const DC&>(), py::arg("A"));
        linres.def(py::init<const LOp&, const DC&>(), py::arg("A"), py::arg("b"));
        linres.def("evaluate", py::overload_cast<const DC&>(&LinRes::evaluate, py::const_),
                   py::arg("x"), py::return_value_policy::move);
        linres.def("getJacobian", py::overload_cast<const DC&>(&LinRes::getJacobian), py::arg("x"),
                   py::return_value_policy::move);
        linres.def("getDomainDescriptor",
                   py::overload_cast<>(&LinRes::getDomainDescriptor, py::const_),
                   py::return_value_policy::reference_internal);
        linres.def("getRangeDescriptor",
                   py::overload_cast<>(&LinRes::getRangeDescriptor, py::const_),
                   py::return_value_policy::reference_internal);
        linres.def("evaluate", py::overload_cast<const DC&, DC&>(&LinRes::evaluate, py::const_),
                   py::arg("x"), py::arg("result"));
    }
} // namespace detail

void add_linear_residual(py::module& m)
{
    detail::add_linear_residual<float>(m, "LinearResidualf");
    detail::add_linear_residual<double>(m, "LinearResiduald");
    detail::add_linear_residual<thrust::complex<float>>(m, "LinearResidualcf");
    detail::add_linear_residual<thrust::complex<double>>(m, "LinearResidualcd");

    m.attr("LinearResidual") = m.attr("LinearResidualf");
}

namespace detail
{
    template <class data_t>
    void add_functional_clonable(py::module& m, const char* name)
    {
        using Cloneable = elsa::Cloneable<elsa::Functional<data_t>>;
        using Functional = elsa::Functional<data_t>;

        py::class_<Cloneable> cloneable(m, name);
        cloneable
            .def("__ne__", py::overload_cast<const Functional&>(&Cloneable::operator!=, py::const_),
                 py::arg("other"))
            .def("__eq__", py::overload_cast<const Functional&>(&Cloneable::operator==, py::const_),
                 py::arg("other"))
            .def("clone", py::overload_cast<>(&Cloneable::clone, py::const_));
    }

    template <class data_t>
    void add_functional(py::module& m, const char* name)
    {
        using Cloneable = elsa::Cloneable<elsa::Functional<data_t>>;
        using Functional = elsa::Functional<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;

        py::class_<Functional, Cloneable> fn(m, name);
        fn.def("getDomainDescriptor",
               py::overload_cast<>(&Functional::getDomainDescriptor, py::const_),
               py::return_value_policy::reference_internal);
        fn.def("evaluate", py::overload_cast<const DataContainer&>(&Functional::evaluate),
               py::arg("x"));
        fn.def("getGradient", py::overload_cast<const DataContainer&>(&Functional::getGradient),
               py::arg("x"), py::return_value_policy::move);
        fn.def("getGradient",
               py::overload_cast<const DataContainer&, DataContainer&>(&Functional::getGradient),
               py::arg("x"), py::arg("result"));
        fn.def("getHessian", py::overload_cast<const DataContainer&>(&Functional::getHessian),
               py::arg("x"), py::return_value_policy::move);

        fn.def(py::self + py::self, py::return_value_policy::take_ownership);
        fn.def(data_t{} * py::self, py::return_value_policy::take_ownership);
        fn.def(py::self * data_t{}, py::return_value_policy::take_ownership);
    }

    template <class data_t>
    void add_functional_add(py::module& m, const char* name)
    {
        using Functional = elsa::Functional<data_t>;

        py::class_<elsa::FunctionalSum<data_t>, Functional> fn(m, name);
        fn.def(py::init<const Functional&, const Functional&>());
    }

    template <class data_t>
    void add_functional_scalar_mul(py::module& m, const char* name)
    {
        using Functional = elsa::Functional<data_t>;

        py::class_<elsa::FunctionalScalarMul<data_t>, Functional> fn(m, name);
        fn.def(py::init<const Functional&, data_t>());
    }
} // namespace detail

void add_functional(py::module& m)
{
    detail::add_functional_clonable<float>(m, "CloneableFunctionalf");
    detail::add_functional_clonable<double>(m, "CloneableFunctionald");
    detail::add_functional_clonable<thrust::complex<float>>(m, "CloneableFunctionalcf");
    detail::add_functional_clonable<thrust::complex<double>>(m, "CloneableFunctionalcd");

    detail::add_functional<float>(m, "Functionalf");
    detail::add_functional<double>(m, "Functionald");
    detail::add_functional<thrust::complex<float>>(m, "Functionalcf");
    detail::add_functional<thrust::complex<double>>(m, "Functionalcd");

    m.attr("Functional") = m.attr("Functionalf");

    detail::add_functional_scalar_mul<float>(m, "FunctionalScalarMulf");
    detail::add_functional_scalar_mul<double>(m, "FunctionalScalarMuld");
    detail::add_functional_scalar_mul<elsa::complex<float>>(m, "FunctionalScalarMulcf");
    detail::add_functional_scalar_mul<elsa::complex<double>>(m, "FunctionalScalarMulcd");
    m.attr("FunctionalScalarMul") = m.attr("FunctionalScalarMulf");

    detail::add_functional_add<float>(m, "FunctionalAddf");
    detail::add_functional_add<double>(m, "FunctionalAddd");
    detail::add_functional_add<elsa::complex<float>>(m, "FunctionalAddcf");
    detail::add_functional_add<elsa::complex<double>>(m, "FunctionalAddcd");
    m.attr("FunctionalAdd") = m.attr("FunctionalAddf");
}

namespace detail
{
    template <template <class> class Fn, class data_t>
    void add_norm(py::module& m, const char* name)
    {
        using Functional = elsa::Functional<data_t>;

        py::class_<Fn<data_t>, Functional> norm(m, name);

        norm.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"));
    }
} // namespace detail

template <template <class> class Fn>
void add_norm(py::module& m, std::string str)
{
    detail::add_norm<Fn, float>(m, fmt::format("{}f", str).c_str());
    detail::add_norm<Fn, double>(m, fmt::format("{}d", str).c_str());
    detail::add_norm<Fn, thrust::complex<float>>(m, fmt::format("{}cf", str).c_str());
    detail::add_norm<Fn, thrust::complex<double>>(m, fmt::format("{}cd", str).c_str());

    m.attr(str.c_str()) = m.attr(fmt::format("{}f", str).c_str());
}

namespace detail
{
    template <class data_t>
    void add_least_squares(py::module& m, const char* name)
    {
        using L2Norm = elsa::LeastSquares<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<L2Norm, Functional> norm(m, name);
        norm.def(py::init<const LOp&, const DataContainer&>(), py::arg("A"), py::arg("b"));
    }

    template <class data_t>
    void add_l2squared(py::module& m, const char* name)
    {
        using L2Norm = elsa::L2Squared<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;
        using DataDesc = elsa::DataDescriptor;
        using Functional = elsa::Functional<data_t>;

        py::class_<L2Norm, Functional> norm(m, name);
        norm.def(py::init<const DataDesc&>(), py::arg("desc"));
        norm.def(py::init<const DataContainer&>(), py::arg("b"));
    }

    template <class data_t>
    void add_l2reg(py::module& m, const char* name)
    {
        using L2Norm = elsa::L2Reg<data_t>;
        using DataDesc = elsa::DataDescriptor;
        using Functional = elsa::Functional<data_t>;

        py::class_<L2Norm, Functional> norm(m, name);
        norm.def(py::init<const DataDesc&>(), py::arg("desc"));
        norm.def(py::init<const elsa::LinearOperator<data_t>&>(), py::arg("A"));
    }
} // namespace detail

void add_l2norm(py::module& m)
{
    detail::add_least_squares<float>(m, "LeastSquaresf");
    detail::add_least_squares<double>(m, "LeastSquaresd");

    m.attr("LeastSquares") = m.attr("LeastSquaresf");

    detail::add_l2squared<float>(m, "L2Squaredf");
    detail::add_l2squared<double>(m, "L2Squaredd");
    detail::add_l2squared<thrust::complex<float>>(m, "L2Squaredcf");
    detail::add_l2squared<thrust::complex<double>>(m, "L2Squaredcd");

    m.attr("L2Squared") = m.attr("L2Squaredf");

    detail::add_l2reg<float>(m, "L2Regf");
    detail::add_l2reg<double>(m, "L2Regd");
    detail::add_l2reg<thrust::complex<float>>(m, "L2Regcf");
    detail::add_l2reg<thrust::complex<double>>(m, "L2Regcd");
    m.attr("L2Reg") = m.attr("L2Regf");
}

namespace detail
{
    template <class data_t>
    void add_weighted_l2norm(py::module& m, const char* name)
    {
        using WL2Norm = elsa::WeightedL2Squared<data_t>;
        using DC = elsa::DataContainer<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<WL2Norm, Functional> norm(m, name);
        norm.def("getWeightingOperator",
                 py::overload_cast<>(&WL2Norm::getWeightingOperator, py::const_),
                 py::return_value_policy::move);
        norm.def(py::init<const DC&>(), py::arg("weights"));
    }

    template <class data_t>
    void add_weighted_leastsquares(py::module& m, const char* name)
    {
        using WL2Norm = elsa::WeightedLeastSquares<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<WL2Norm, Functional> norm(m, name);
        norm.def(py::init<const LOp&, const DataContainer&, const DataContainer&>(), py::arg("A"),
                 py::arg("b"), py::arg("weights"));
    }
} // namespace detail

void add_weighted_l2norm(py::module& m)
{
    detail::add_weighted_l2norm<float>(m, "WeightedL2Squaredf");
    detail::add_weighted_l2norm<double>(m, "WeightedL2Squaredd");
    detail::add_weighted_l2norm<thrust::complex<float>>(m, "WeightedL2Squaredcf");
    detail::add_weighted_l2norm<thrust::complex<double>>(m, "WeightedL2Squaredcd");

    m.attr("WeightedL2Squared") = m.attr("WeightedL2Squaredf");
}

namespace detail
{
    template <class data_t>
    void add_weighted_l1norm(py::module& m, const char* name)
    {
        using WL1Norm = elsa::WeightedL1Norm<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<WL1Norm, Functional> norm(m, name);
        norm.def("getWeightingOperator",
                 py::overload_cast<>(&WL1Norm::getWeightingOperator, py::const_),
                 py::return_value_policy::reference_internal);
        norm.def(py::init<const elsa::DataContainer<data_t>&>(), py::arg("weights"));
    }
} // namespace detail

void add_weighted_l1norm(py::module& m)
{
    detail::add_weighted_l1norm<float>(m, "WeightedL1Normf");
    detail::add_weighted_l1norm<double>(m, "WeightedL1Normd");

    m.attr("WeightedL1Norm") = m.attr("WeightedL1Normf");
}

namespace detail
{
    template <class data_t>
    void add_huber_norm(py::module& m, const char* name)
    {
        using Huber = elsa::Huber<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<Huber, Functional> norm(m, name);
        norm.def(py::init<const elsa::DataDescriptor&, data_t>(), py::arg("domainDescriptor"),
                 py::arg("delta") = data_t{1e-06});
    }
} // namespace detail

void add_huber_norm(py::module& m)
{
    detail::add_huber_norm<float>(m, "Huberf");
    detail::add_huber_norm<double>(m, "Huberd");

    m.attr("Huber") = m.attr("Huberf");
}

namespace detail
{
    template <class data_t>
    void add_pseudohuber_norm(py::module& m, const char* name)
    {
        using PseudoHuber = elsa::PseudoHuber<data_t>;
        using Functional = elsa::Functional<data_t>;

        py::class_<PseudoHuber, Functional> norm(m, name);
        norm.def(py::init<const elsa::DataDescriptor&, data_t>(), py::arg("domainDescriptor"),
                 py::arg("delta") = static_cast<data_t>(1e-06));
    }
} // namespace detail

void add_pseudohuber_norm(py::module& m)
{
    detail::add_pseudohuber_norm<float>(m, "PseudoHuberf");
    detail::add_pseudohuber_norm<double>(m, "PseudoHuberd");

    m.attr("PseudoHuber") = m.attr("PseudoHuberf");
}

namespace detail
{
    template <class data_t>
    void add_quadric_fn(py::module& m, const char* name)
    {
        using Fn = elsa::Quadric<data_t>;
        using Functional = elsa::Functional<data_t>;

        using DataContainer = elsa::DataContainer<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<Fn, Functional> fn(m, name);
        fn.def("getGradientExpression", py::overload_cast<>(&Fn::getGradientExpression, py::const_),
               py::return_value_policy::reference_internal);
        fn.def(py::init<const DataContainer&>(), py::arg("b"));
        fn.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"));
        fn.def(py::init<const LOp&>(), py::arg("A"));
        fn.def(py::init<const LOp&, const DataContainer&>(), py::arg("A"), py::arg("b"));
    }
} // namespace detail

void add_quadric(py::module& m)
{
    detail::add_quadric_fn<float>(m, "Quadricf");
    detail::add_quadric_fn<double>(m, "Quadricd");
    detail::add_quadric_fn<thrust::complex<float>>(m, "Quadriccf");
    detail::add_quadric_fn<thrust::complex<double>>(m, "Quadriccd");

    m.attr("Quadric") = m.attr("Quadricf");
}

namespace detail
{
    template <class data_t>
    void add_emission_log_fn(py::module& m, const char* name)
    {
        using Fn = elsa::EmissionLogLikelihood<data_t>;
        using Functional = elsa::Functional<data_t>;

        using DC = elsa::DataContainer<data_t>;
        using DD = elsa::DataDescriptor;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<Fn, Functional> fn(m, name);
        fn.def(py::init<const LOp&, const DC&>(), py::arg("A"), py::arg("y"));
        fn.def(py::init<const LOp&, const DC&, const DC&>(), py::arg("A"), py::arg("y"),
               py::arg("r"));
    }
} // namespace detail

void add_emissionlog(py::module& m)
{
    detail::add_emission_log_fn<float>(m, "EmissionLogLikelihoodf");
    detail::add_emission_log_fn<double>(m, "EmissionLogLikelihoodd");

    m.attr("EmissionLogLikelihood") = m.attr("EmissionLogLikelihoodf");
}

namespace detail
{
    template <class data_t>
    void add_transmission_log_fn(py::module& m, const char* name)
    {
        using Fn = elsa::TransmissionLogLikelihood<data_t>;
        using Functional = elsa::Functional<data_t>;

        using DC = elsa::DataContainer<data_t>;
        using DD = elsa::DataDescriptor;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<Fn, Functional> fn(m, name);
        fn.def(py::init<const LOp&, const DC&, const DC&>(), py::arg("A"), py::arg("y"),
               py::arg("b"));
        fn.def(py::init<const LOp&, const DC&, const DC&, const DC&>(), py::arg("A"), py::arg("y"),
               py::arg("b"), py::arg("r"));
    }
} // namespace detail

void add_transmission(py::module& m)
{
    detail::add_transmission_log_fn<float>(m, "TransmissionLogLikelihoodf");
    detail::add_transmission_log_fn<double>(m, "TransmissionLogLikelihoodd");

    m.attr("TransmissionLogLikelihood") = m.attr("TransmissionLogLikelihoodf");
}

namespace detail
{
    template <class data_t>
    void add_constraint_clonable(py::module& m, const char* name)
    {
        using Constraint = elsa::Constraint<data_t>;
        using Cloneable = elsa::Cloneable<Constraint>;

        py::class_<Cloneable> cloneable(m, name);
        cloneable
            .def("__ne__", py::overload_cast<const Constraint&>(&Cloneable::operator!=, py::const_),
                 py::arg("other"))
            .def("__eq__",
                 py::overload_cast<const Constraint&>(&Constraint::operator==, py::const_),
                 py::arg("other"))
            .def("clone", py::overload_cast<>(&Cloneable::clone, py::const_));
    }

    template <class data_t>
    void add_constraint(py::module& m, const char* name)
    {
        using Constraint = elsa::Constraint<data_t>;
        using Cloneable = elsa::Cloneable<Constraint>;

        using LOp = elsa::LinearOperator<data_t>;

        auto ref_internal = py::return_value_policy::reference_internal;

        py::class_<Constraint, Cloneable> constraint(m, name);
        constraint.def("getDataVectorC",
                       py::overload_cast<>(&Constraint::getDataVectorC, py::const_), ref_internal);
        constraint.def("getOperatorA", py::overload_cast<>(&Constraint::getOperatorA, py::const_),
                       ref_internal);
        constraint.def("getOperatorB", py::overload_cast<>(&Constraint::getOperatorB, py::const_),
                       ref_internal);
        constraint.def(py::init<const LOp&, const LOp&, const elsa::DataContainer<data_t>&>(),
                       py::arg("A"), py::arg("B"), py::arg("c"));
    }
} // namespace detail

void add_constraint(py::module& m)
{
    detail::add_constraint_clonable<float>(m, "CloneableConstraintf");
    detail::add_constraint_clonable<double>(m, "CloneableConstraintd");
    detail::add_constraint_clonable<thrust::complex<float>>(m, "CloneableConstraintcf");
    detail::add_constraint_clonable<thrust::complex<double>>(m, "CloneableConstraintcd");

    m.attr("CloneableConstraint") = m.attr("CloneableConstraintf");

    detail::add_constraint<float>(m, "Constraintf");
    detail::add_constraint<double>(m, "Constraintd");
    detail::add_constraint<thrust::complex<float>>(m, "Constraintcf");
    detail::add_constraint<thrust::complex<double>>(m, "Constraintcd");

    m.attr("Constraint") = m.attr("Constraintf");
}

namespace detail
{
    template <class data_t>
    void add_axdtStatRecon_fn(py::module& m, const char* name)
    {
        using Fn = elsa::AXDTStatRecon<data_t>;
        using Functional = elsa::Functional<data_t>;

        using DC = elsa::DataContainer<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<Fn, Functional> fn(m, name);
        py::enum_<typename Fn::StatReconType>(fn, "StatReconType")
            .value("Gaussian_log_d", Fn::StatReconType::Gaussian_log_d)
            .value("Gaussian_d", Fn::StatReconType::Gaussian_d)
            .value("Gaussian_approximate_racian", Fn::StatReconType::Gaussian_approximate_racian)
            .value("Racian_direct", Fn::StatReconType::Racian_direct)
            .export_values();

        fn.def(py::init<const DC&, const DC&, const DC&, const DC&,
                        const LOp&, const LOp&, long, const typename Fn::StatReconType&>(),
               py::arg("ffa"), py::arg("ffb"), py::arg("a"), py::arg("b"),
               py::arg("absorp_op"), py::arg("axdt_op"), py::arg("N"), py::arg("recon_type"));

        fn.def(py::init<const DC&, const LOp&, const typename Fn::StatReconType&>(),
               py::arg("axdt_proj"), py::arg("axdt_op"), py::arg("recon_type"));
    }
} // namespace detail

void add_AXDTStatRecon(py::module& m)
{
    detail::add_axdtStatRecon_fn<float>(m, "AXDTStatReconf");
    detail::add_axdtStatRecon_fn<double>(m, "AXDTStatRecond");

    m.attr("AXDTStatRecon") = m.attr("AXDTStatReconf");
}

namespace detail
{
    template <class data_t>
    void add_separable_sum_clonable(py::module& m, const char* name)
    {
        using T = elsa::SeparableSum<data_t>;
        using Clone = elsa::Cloneable<T>;

        py::class_<Clone> cloneable(m, name);
        cloneable
            .def("__ne__", py::overload_cast<const T&>(&Clone::operator!=, py::const_),
                 py::arg("other"))
            .def("__eq__", py::overload_cast<const T&>(&Clone::operator==, py::const_),
                 py::arg("other"))
            .def("clone", py::overload_cast<>(&Clone::clone, py::const_));
    }

    template <class data_t>
    void add_separable_sum(py::module& m, const char* name)
    {
        using T = elsa::SeparableSum<data_t>;
        using Functional = elsa::Functional<data_t>;

        using LOp = elsa::LinearOperator<data_t>;

        auto ref_internal = py::return_value_policy::reference_internal;

        py::class_<T, Functional> sepsum(m, name);
        sepsum.def(py::init<const Functional&>());
        sepsum.def(py::init<const Functional&, const Functional&>());
        sepsum.def(py::init<const Functional&, const Functional&, const Functional&>());
        sepsum.def(
            py::init<const Functional&, const Functional&, const Functional&, const Functional&>());
    }
} // namespace detail

void add_separable_sum(py::module& m)
{
    detail::add_separable_sum_clonable<float>(m, "CloneableSeparableSumf");
    detail::add_separable_sum_clonable<double>(m, "CloneableSeparableSumd");

    m.attr("CloneableSeparableSum") = m.attr("CloneableSeparableSumf");

    detail::add_separable_sum<float>(m, "SeparableSumf");
    detail::add_separable_sum<double>(m, "SeparableSumd");

    m.attr("SeparableSum") = m.attr("SeparableSumf");
}

void add_definitions_pyelsa_functionals(py::module& m)
{
    add_linear_residual(m);
    add_functional(m);

    add_norm<elsa::L0PseudoNorm>(m, "L0PseudoNorm");
    add_norm<elsa::L1Norm>(m, "L1Norm");
    add_norm<elsa::LInfNorm>(m, "LInfNorm");

    add_l2norm(m);
    add_weighted_l2norm(m);
    add_weighted_l1norm(m);

    add_huber_norm(m);
    add_pseudohuber_norm(m);

    add_quadric(m);

    add_emissionlog(m);
    add_transmission(m);

    add_constraint(m);
    add_separable_sum(m);

    add_AXDTStatRecon(m);

    elsa::FunctionalsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_functionals, m)
{
    add_definitions_pyelsa_functionals(m);
}
