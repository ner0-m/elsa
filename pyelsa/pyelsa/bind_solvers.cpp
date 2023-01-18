#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "CG.h"
#include "FGM.h"
#include "PGD.h"
#include "FISTA.h"
#include "GradientDescent.h"
#include "Landweber.h"
#include "SIRT.h"
#include "OGM.h"
#include "OrthogonalMatchingPursuit.h"
#include "SQS.h"
#include "Solver.h"

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
        using Problem = elsa::Problem<data_t>;

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
    void add_conjugate_gradient(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Problem<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::CG<data_t>, Solver> cg(m, name);
        cg.def(py::init<const Problem&, const LOp&>(), py::arg("problem"),
               py::arg("preconditionerInverse"));
        cg.def(py::init<const Problem&, const LOp&, data_t>(), py::arg("problem"),
               py::arg("preconditionerInverse"), py::arg("epsilon"));
        cg.def(py::init<const Problem&>(), py::arg("problem"));
        cg.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
    }
} // namespace detail

void add_conjugate_gradient(py::module& m)
{
    detail::add_conjugate_gradient<float>(m, "CGf");
    detail::add_conjugate_gradient<double>(m, "CGd");

    m.attr("CG") = m.attr("CGf");
}

namespace detail
{
    template <class data_t>
    void add_ista(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Problem<data_t>;
        using Threshold = elsa::geometry::Threshold<data_t>;

        py::class_<elsa::PGD<data_t>, Solver> ista(m, name);
        ista.def(py::init<const Problem&, Threshold>(), py::arg("problem"), py::arg("mu"));
        ista.def(py::init<const Problem&, Threshold, data_t>(), py::arg("problem"), py::arg("mu"),
                 py::arg("epsilon"));
        ista.def(py::init<const Problem&>(), py::arg("problem"));
        ista.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
    }
} // namespace detail

void add_ista(py::module& m)
{
    detail::add_ista<float>(m, "PGDf");
    detail::add_ista<double>(m, "PGDd");

    m.attr("PGD") = m.attr("PGDf");
}

namespace detail
{
    template <class data_t>
    void add_fista(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Problem<data_t>;
        using Threshold = elsa::geometry::Threshold<data_t>;

        py::class_<elsa::FISTA<data_t>, Solver> fista(m, name);
        fista.def(py::init<const Problem&, Threshold>(), py::arg("problem"), py::arg("mu"));
        fista.def(py::init<const Problem&, Threshold, data_t>(), py::arg("problem"), py::arg("mu"),
                  py::arg("epsilon"));
        fista.def(py::init<const Problem&>(), py::arg("problem"));
        fista.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
    }
} // namespace detail

void add_fista(py::module& m)
{
    detail::add_fista<float>(m, "FISTAf");
    detail::add_fista<double>(m, "FISTAd");

    m.attr("FISTA") = m.attr("FISTAf");
}

namespace detail
{
    template <class data_t>
    void add_fgm(py::module& m, const char* name)
    {
        using Solver = elsa::Solver<data_t>;
        using Problem = elsa::Problem<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::FGM<data_t>, Solver> fgm(m, name);
        fgm.def(py::init<const Problem&, const LOp&>(), py::arg("problem"),
                py::arg("preconditionerInverse"));
        fgm.def(py::init<const Problem&, const LOp&, data_t>(), py::arg("problem"),
                py::arg("preconditionerInverse"), py::arg("epsilon"));
        fgm.def(py::init<const Problem&>(), py::arg("problem"));
        fgm.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
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
        using Problem = elsa::Problem<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::OGM<data_t>, Solver> ogm(m, name);
        ogm.def(py::init<const Problem&, const LOp&>(), py::arg("problem"),
                py::arg("preconditionerInverse"));
        ogm.def(py::init<const Problem&, const LOp&, data_t>(), py::arg("problem"),
                py::arg("preconditionerInverse"), py::arg("epsilon"));
        ogm.def(py::init<const Problem&>(), py::arg("problem"));
        ogm.def(py::init<const Problem&, data_t>(), py::arg("problem"), py::arg("epsilon"));
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
        using Problem = elsa::Problem<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::SQS<data_t>, Solver> sqs(m, name);
        sqs.def(py::init<const Problem&, bool>(), py::arg("problem"),
                py::arg("momentumAcceleration") = static_cast<bool>(true));
        sqs.def(py::init<const Problem&, bool, data_t>(), py::arg("problem"),
                py::arg("momentumAcceleration"), py::arg("epsilon"));
        sqs.def(py::init<const Problem&, const LOp&, bool>(), py::arg("problem"),
                py::arg("preconditioner"),
                py::arg("momentumAcceleration") = static_cast<bool>(true));
        sqs.def(py::init<const Problem&, LOp&, bool, data_t>(), py::arg("problem"),
                py::arg("preconditioner"), py::arg("momentumAcceleration"), py::arg("epsilon"));
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
        using Problem = elsa::Problem<data_t>;
        using LOp = elsa::LinearOperator<data_t>;

        py::class_<elsa::OrthogonalMatchingPursuit<data_t>, Solver> omp(m, name);
        omp.def(py::init<const elsa::RepresentationProblem<data_t>&, data_t>(), py::arg("problem"),
                py::arg("epsilon"));
    }
} // namespace detail

void add_omp(py::module& m)
{
    detail::add_omp<float>(m, "OrthogonalMatchingPursuitf");
    detail::add_omp<double>(m, "OrthogonalMatchingPursuitd");

    m.attr("OrthogonalMatchingPursuit") = m.attr("OrthogonalMatchingPursuitf");
}

void add_definitions_pyelsa_solvers(py::module& m)
{
    add_solver(m);
    add_gradient_descent(m);
    add_conjugate_gradient(m);
    add_ista(m);
    add_fista(m);

    add_fgm(m);
    add_ogm(m);
    add_sqs(m);
    add_omp(m);

    py::class_<elsa::Landweber<float>, elsa::Solver<float>> Landweberf(m, "Landweberf");
    Landweberf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
        py::arg("A"), py::arg("b"));
    Landweberf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&, float>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));
    Landweberf.def(py::init<const elsa::WLSProblem<float>&>(), py::arg("problem"));
    Landweberf.def(py::init<const elsa::WLSProblem<float>&, float>(), py::arg("problem"),
                   py::arg("stepSize"));

    m.attr("Landweber") = m.attr("Landweberf");

    py::class_<elsa::Landweber<double>, elsa::Solver<double>> Landweberd(m, "Landweberd");
    Landweberd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
        py::arg("A"), py::arg("b"));
    Landweberd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&, double>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));
    Landweberd.def(py::init<const elsa::WLSProblem<double>&>(), py::arg("problem"));
    Landweberd.def(py::init<const elsa::WLSProblem<double>&, double>(), py::arg("problem"),
                   py::arg("stepSize"));

    py::class_<elsa::SIRT<float>, elsa::Solver<float>> sirtf(m, "SIRTf");
    sirtf.def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
              py::arg("A"), py::arg("b"));
    sirtf.def(
        py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&, float>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));
    sirtf.def(py::init<const elsa::WLSProblem<float>&>(), py::arg("problem"));
    sirtf.def(py::init<const elsa::WLSProblem<float>&, float>(), py::arg("problem"),
              py::arg("stepSize"));

    m.attr("SIRT") = m.attr("SIRTf");

    py::class_<elsa::SIRT<double>, elsa::Solver<double>> sirtd(m, "SIRTd");
    sirtd.def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
              py::arg("A"), py::arg("b"));
    sirtd.def(
        py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&, double>(),
        py::arg("A"), py::arg("b"), py::arg("stepSize"));
    sirtd.def(py::init<const elsa::WLSProblem<double>&>(), py::arg("problem"));
    sirtd.def(py::init<const elsa::WLSProblem<double>&, double>(), py::arg("problem"),
              py::arg("stepSize"));

    elsa::SolversHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_solvers, m)
{
    add_definitions_pyelsa_solvers(m);
}
