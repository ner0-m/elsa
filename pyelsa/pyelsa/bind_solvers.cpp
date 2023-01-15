#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Dictionary.h"
#include "FGM.h"
#include "LeastSquares.h"
#include "LinearOperator.h"
#include "PGD.h"
#include "APGD.h"
#include "ADMML2.h"
#include "GradientDescent.h"
#include "Landweber.h"
#include "ProximalOperator.h"
#include "SIRT.h"
#include "OGM.h"
#include "CGLS.h"
#include "CGNonlinear.h"
#include "CGNE.h"
#include "OrthogonalMatchingPursuit.h"
#include "SQS.h"
#include "AB_GMRES.h"
#include "BA_GMRES.h"
#include "LinearizedADMM.h"
#include "Solver.h"
#include "ProximalOperator.h"

#include "bind_common.h"
#include "hints/solvers_hints.cpp"

namespace py = pybind11;

template <class data_t>
void add_definitions_solver(
    py::class_<elsa::Solver<data_t>, elsa::Cloneable<elsa::Solver<data_t>>> solver)
{
}

namespace detail
{
    template <class data_t>
    void add_solver_cloneable(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Cloneable = elsa::Cloneable<Solver>;

        py::class_<Cloneable> cloneable(m, name);
        cloneable
            .def("__ne__", py::overload_cast<const Solver&>(&Cloneable::operator!=, py::const_),
                 py::arg("other"))
            .def("__eq__", py::overload_cast<const Solver&>(&Cloneable::operator==, py::const_),
                 py::arg("other"))
            .def("clone", py::overload_cast<>(&Cloneable::clone, py::const_));
    }

    template <class data_t>
    void add_solver(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Cloneable = elsa::Cloneable<Solver>;

        py::class_<Solver, Cloneable> solver(m, name);
        solver.def("solve",
                   py::overload_cast<long, std::optional<elsa::DataContainer<data_t>>>(
                       &elsa::Solver<data_t>::solve),
                   py::arg("iters"), py::arg("x0") = py::none(), py::return_value_policy::move);
    }
} // namespace detail

void add_solver(py::module& m)
{
    detail::add_solver_cloneable<float>(m, "CloneableSolverf");
    detail::add_solver_cloneable<double>(m, "CloneableSolverd");
    detail::add_solver_cloneable<thrust::complex<float>>(m, "CloneableSolvercf");
    detail::add_solver_cloneable<thrust::complex<double>>(m, "CloneableSolvercd");

    detail::add_solver<float>(m, "Solverf");
    detail::add_solver<double>(m, "Solverd");
    detail::add_solver<thrust::complex<float>>(m, "Solvercf");
    detail::add_solver<thrust::complex<double>>(m, "Solvercd");

    m.attr("Solver") = m.attr("Solverf");
}

namespace detail
{
    template <class data_t>
    void add_gradient_descent(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Functional<data_t>;

        py::class_<elsa::GradientDescent<data_t>, Solver> solver(m, name);
        solver.def(py::init<const Problem&>(), py::arg("problem"));
        solver.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("stepSize"));
    }
} // namespace detail

void add_gradient_descent(py::module& m)
{
    detail::add_gradient_descent<float>(m, "GradientDescentf");
    detail::add_gradient_descent<double>(m, "GradientDescentd");

    m.attr("GradientDescent") = m.attr("GradientDescentf");
}

namespace detail
{
    template <class data_t>
    void add_cgls(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::CGLS<data_t>, Solver> cg(m, name);
        cg.def(py::init<const LOp&, const elsa::DataContainer<data_t>&, data_t>(), py::arg("A"),
               py::arg("b"), py::arg("eps") = 0.0);
    }
} // namespace detail

void add_cgls(py::module& m)
{
    detail::add_cgls<float>(m, "CGLSf");
    detail::add_cgls<double>(m, "CGLSd");

    m.attr("CGLS") = m.attr("CGLSf");
}

namespace detail
{
    template <class data_t>
    void add_cgne(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::CGNE<data_t>, Solver> cg(m, name);
        cg.def(py::init<const LOp&, const elsa::DataContainer<data_t>&>(), py::arg("A"),
               py::arg("b"));
    }
} // namespace detail

void add_cgne(py::module& m)
{
    detail::add_cgne<float>(m, "CGNEf");
    detail::add_cgne<double>(m, "CGNEd");

    m.attr("CGNE") = m.attr("CGNEf");
}

namespace detail
{
    template <class data_t>
    void add_nonlinear_conjugate_gradient(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Problem<data_t>;

        py::class_<elsa::CG<data_t>, Solver> cg(m, name);
        cg.def(py::init<const Problem&>(), py::arg("problem"));
        cg.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
    }
} // namespace detail

void add_nonlinear_conjugate_gradient(py::module& m)
{
    detail::add_nonlinear_conjugate_gradient<float>(m, "CGNonlinearf");
    detail::add_nonlinear_conjugate_gradient<double>(m, "CGNonlieard");

    m.attr("CGNonlinear") = m.attr("CGNonlinearf");
}

namespace detail
{
    template <class data_t>
    void add_ista(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using DC = elsa::DataContainer<data_t>;
        using Prox = elsa::ProximalOperator<data_t>;

        py::class_<elsa::PGD<data_t>, Solver> pgd(m, name);
        pgd.def(py::init<const LOp&, const DC&, Prox, std::optional<data_t>, data_t>(),
                py::arg("A"), py::arg("b"), py::arg("prox"), py::arg("mu") = py::none(),
                py::arg("eps") = 1e-6);
    }
} // namespace detail

void add_ista(py::module& m)
{
    detail::add_ista<float>(m, "PGDf");
    detail::add_ista<double>(m, "PGDd");

    m.attr("PGD") = m.attr("PGDf");
    m.attr("ISTA") = m.attr("PGDf");
}

namespace detail
{
    template <class data_t>
    void add_fista(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using Prox = elsa::ProximalOperator<data_t>;
        using DC = elsa::DataContainer<data_t>;

        py::class_<elsa::APGD<data_t>, Solver> apgd(m, name);
        apgd.def(py::init<const LOp&, const DC&, Prox, std::optional<data_t>, data_t>(),
                 py::arg("A"), py::arg("b"), py::arg("prox"), py::arg("mu") = py::none(),
                 py::arg("eps") = 1e-6);
    }
} // namespace detail

void add_fista(py::module& m)
{
    detail::add_fista<float>(m, "APGDf");
    detail::add_fista<double>(m, "APGDd");

    m.attr("APGD") = m.attr("APGDf");
    m.attr("FISTA") = m.attr("APGDf");
}

namespace detail
{
    template <class data_t>
    void add_fgm(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Functional<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::FGM<data_t>, Solver> fgm(m, name);
        fgm.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("eps") = 1e-10);
        fgm.def(py::init<const Problem&, const LOp&, data_t>(), py::arg("problem"),
                py::arg("preconditioner"), py::arg("eps") = 1e-10);
    }
} // namespace detail

void add_fgm(py::module& m)
{
    detail::add_fgm<float>(m, "FGMf");
    detail::add_fgm<double>(m, "FMGd");

    m.attr("FGM") = m.attr("FGMf");
}

namespace detail
{
    template <class data_t>
    void add_ogm(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Functional<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::OGM<data_t>, Solver> ogm(m, name);
        ogm.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("eps") = 1e-10);
        ogm.def(py::init<const Problem&, const LOp&, data_t>(), py::arg("problem"),
                py::arg("preconditioner"), py::arg("eps") = 1e-10);
    }
} // namespace detail

void add_ogm(py::module& m)
{
    detail::add_ogm<float>(m, "OGMf");
    detail::add_ogm<double>(m, "OGMd");

    m.attr("OGM") = m.attr("OGMf");
}

namespace detail
{
    template <class data_t>
    void add_sqs(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using L2Pow2 = elsa::LeastSquares<data_t>;
        using Subsets = std::vector<std::unique_ptr<L2Pow2>>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::SQS<data_t>, Solver> sqs(m, name);
        sqs.def(py::init<const L2Pow2&, bool, data_t>(), py::arg("problem"),
                py::arg("momentumAcceleration") = static_cast<bool>(true), py::arg("eps") = 1e-10);
        sqs.def(py::init<const L2Pow2&, const LOp&, bool, data_t>(), py::arg("problem"),
                py::arg("preconditioner"),
                py::arg("momentumAcceleration") = static_cast<bool>(true), py::arg("eps") = 1e-10);

        // TODO: This should be possible
        // sqs.def(py::init<const L2Pow2&, Subsets&&, bool, data_t>(), py::arg("problem"),
        //         py::arg("subsets"), py::arg("momentumAcceleration") = static_cast<bool>(true),
        //         py::arg("eps") = 1e-10);
        // sqs.def(py::init<const L2Pow2&, Subsets&&, const LOp&, bool, data_t>(),
        // py::arg("problem"),
        //         py::arg("subsets"), py::arg("preconditioner"),
        //         py::arg("momentumAcceleration") = static_cast<bool>(true), py::arg("eps") =
        //         1e-10);
    }
} // namespace detail

void add_sqs(py::module& m)
{
    detail::add_sqs<float>(m, "SQSf");
    detail::add_sqs<double>(m, "SQSd");

    m.attr("SQS") = m.attr("SQSf");
}

namespace detail
{
    template <class data_t>
    void add_omp(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Dict = elsa::Dictionary<data_t>;
        using DC = elsa::DataContainer<data_t>;

        py::class_<elsa::OrthogonalMatchingPursuit<data_t>, Solver> omp(m, name);
        omp.def(py::init<const Dict&, const DC&, data_t>(), py::arg("dict"), py::arg("signal"),
                py::arg("epsilon"));
    }
} // namespace detail

void add_omp(py::module& m)
{
    detail::add_omp<float>(m, "OrthogonalMatchingPursuitf");
    detail::add_omp<double>(m, "OrthogonalMatchingPursuitd");

    m.attr("OrthogonalMatchingPursuit") = m.attr("OrthogonalMatchingPursuitf");
}

namespace detail
{
    template <class data_t, template <class> class solver_t = elsa::Solver>
    void add_generalized_minimum_residual(py::module& m, const char* name)
    {
        using gmres = solver_t<data_t>;
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using dc = elsa::DataContainer<data_t>;

        py::class_<gmres, Solver> GMRES(m, name);
        GMRES
            .def(py::init<const LOp&, const LOp&, const dc&>(), py::arg("projector"),
                 py::arg("backprojector"), py::arg("sinogram"))
            .def(py::init<const LOp&, const LOp&, const dc&, data_t>(), py::arg("projector"),
                 py::arg("backprojector"), py::arg("sinogram"), py::arg("epsilon"))
            .def(py::init<const LOp&, const dc&, data_t>(), py::arg("projector"),
                 py::arg("sinogram"), py::arg("epsilon"))
            .def(py::init<const LOp&, const dc&>(), py::arg("projector"), py::arg("sinogram"));
    }

    template <class data_t>
    void add_admml2(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using DC = elsa::DataContainer<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using Prox = elsa::ProximalOperator<data_t>;

        py::class_<elsa::ADMML2<data_t>, Solver> admm(m, name);
        admm.def(py::init<const LOp&, const DC&, const LOp&, const Prox&, std::optional<data_t>>(),
                 py::arg("Op"), py::arg("b"), py::arg("A"), py::arg("proxg"),
                 py::arg("tau") = py::none());
    }

    template <class data_t>
    void add_ladmm(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using LOp = elsa::LinearOperator<data_t>;
        using ProxOp = elsa::ProximalOperator<data_t>;

        py::class_<elsa::LinearizedADMM<data_t>, Solver> ladmm(m, name);
        ladmm.def(py::init<const LOp&, ProxOp, ProxOp, data_t, data_t, bool>(), py::arg("K"),
                  py::arg("proxf"), py::arg("proxg"), py::arg("sigma"), py::arg("tau"),
                  py::arg("computeKNorm") = true);
    }
} // namespace detail

void add_admml2(py::module& m)
{
    detail::add_admml2<float>(m, "ADMML2f");
    detail::add_admml2<double>(m, "ADMML2d");

    m.attr("ADMML2") = m.attr("ADMML2f");
}

void add_ladmm(py::module& m)
{
    detail::add_ladmm<float>(m, "LinearizedADMMf");
    detail::add_ladmm<double>(m, "LinearizedADMMd");

    m.attr("LinearizedADMM") = m.attr("LinearizedADMMf");
}

void add_generalized_minimum_residual(py::module& m)
{
    detail::add_generalized_minimum_residual<float, elsa::AB_GMRES>(m, "ABGMRESf");
    detail::add_generalized_minimum_residual<double, elsa::AB_GMRES>(m, "ABGMRESd");

    detail::add_generalized_minimum_residual<float, elsa::BA_GMRES>(m, "BAGMRESf");
    detail::add_generalized_minimum_residual<double, elsa::BA_GMRES>(m, "BAGMRESd");

    m.attr("ABGMRES") = m.attr("ABGMRESf");
    m.attr("BAGMRES") = m.attr("BAGMRESf");

    // adding a GMRES function that uses ABGMRES by default:
    m.attr("GMRES") = m.attr("ABGMRESf");
}

void add_definitions_pyelsa_solvers(py::module& m)
{
    add_solver(m);
    add_gradient_descent(m);
    add_cgls(m);
    add_cgne(m);
    add_nonlinear_conjugate_gradient(m);
    add_ista(m);
    add_fista(m);

    add_fgm(m);
    add_ogm(m);
    add_sqs(m);
    add_omp(m);
    add_ladmm(m);
    add_admml2(m);

    add_generalized_minimum_residual(m);

    py::class_<elsa::Landweber<float>, elsa::Solver<float>> Landweberf(m, "Landweberf");
    Landweberf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
        py::arg("A"), py::arg("b"));
    Landweberf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&, float>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));

    m.attr("Landweber") = m.attr("Landweberf");

    py::class_<elsa::Landweber<double>, elsa::Solver<double>> Landweberd(m, "Landweberd");
    Landweberd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
        py::arg("A"), py::arg("b"));
    Landweberd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&, double>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));

    py::class_<elsa::SIRT<float>, elsa::Solver<float>> sirtf(m, "SIRTf");
    sirtf.def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
              py::arg("A"), py::arg("b"));
    sirtf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&, float>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));

    m.attr("SIRT") = m.attr("SIRTf");

    py::class_<elsa::SIRT<double>, elsa::Solver<double>> sirtd(m, "SIRTd");
    sirtd.def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
              py::arg("A"), py::arg("b"));
    sirtd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&, double>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));

    elsa::SolversHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_solvers, m)
{
    add_definitions_pyelsa_solvers(m);
}
