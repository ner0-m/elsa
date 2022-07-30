#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#include "CG.h"
#include "FGM.h"
#include "FISTA.h"
#include "GradientDescent.h"
#include "ISTA.h"
#include "OGM.h"
#include "OrthogonalMatchingPursuit.h"
#include "SQS.h"
#include "Solver.h"

#include "hints/solvers_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_solvers(py::module& m)
{
    py::class_<elsa::Cloneable<elsa::Solver<float>>> CloneableSolverf(m, "CloneableSolverf");
    CloneableSolverf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Solver<float>>::*)(const elsa::Solver<float>&)
                  const)(&elsa::Cloneable<elsa::Solver<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Solver<float>>::*)(const elsa::Solver<float>&)
                  const)(&elsa::Cloneable<elsa::Solver<float>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Solver<float>, std::default_delete<elsa::Solver<float>>>(
                 elsa::Cloneable<elsa::Solver<float>>::*)()
                  const)(&elsa::Cloneable<elsa::Solver<float>>::clone));

    py::class_<elsa::Solver<float>, elsa::Cloneable<elsa::Solver<float>>> Solverf(m, "Solverf");
    Solverf
        .def("solve",
             (elsa::DataContainer<float> & (elsa::Solver<float>::*) (long) )(
                 &elsa::Solver<float>::solve),
             py::arg("iterations") = static_cast<long>(0),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::Solver<float>::*) ())(
                 &elsa::Solver<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::Solver<float>::*) ()
                  const)(&elsa::Solver<float>::getCurrentSolution),
             py::return_value_policy::reference_internal);

    m.attr("Solver") = m.attr("Solverf");

    py::class_<elsa::Cloneable<elsa::Solver<double>>> CloneableSolverd(m, "CloneableSolverd");
    CloneableSolverd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Solver<double>>::*)(const elsa::Solver<double>&)
                  const)(&elsa::Cloneable<elsa::Solver<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Solver<double>>::*)(const elsa::Solver<double>&)
                  const)(&elsa::Cloneable<elsa::Solver<double>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Solver<double>, std::default_delete<elsa::Solver<double>>>(
                 elsa::Cloneable<elsa::Solver<double>>::*)()
                  const)(&elsa::Cloneable<elsa::Solver<double>>::clone));

    py::class_<elsa::Solver<double>, elsa::Cloneable<elsa::Solver<double>>> Solverd(m, "Solverd");
    Solverd
        .def("solve",
             (elsa::DataContainer<double> & (elsa::Solver<double>::*) (long) )(
                 &elsa::Solver<double>::solve),
             py::arg("iterations") = static_cast<long>(0),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::Solver<double>::*) ())(
                 &elsa::Solver<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::Solver<double>::*) ()
                  const)(&elsa::Solver<double>::getCurrentSolution),
             py::return_value_policy::reference_internal);

    py::class_<elsa::Cloneable<elsa::Solver<std::complex<float>>>> CloneableSolvercf(
        m, "CloneableSolvercf");
    CloneableSolvercf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Solver<std::complex<float>>>::*)(
                 const elsa::Solver<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Solver<std::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Solver<std::complex<float>>>::*)(
                 const elsa::Solver<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Solver<std::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Solver<std::complex<float>>,
                                       std::default_delete<elsa::Solver<std::complex<float>>>>(
                          elsa::Cloneable<elsa::Solver<std::complex<float>>>::*)()
                           const)(&elsa::Cloneable<elsa::Solver<std::complex<float>>>::clone));

    py::class_<elsa::Solver<std::complex<float>>,
               elsa::Cloneable<elsa::Solver<std::complex<float>>>>
        Solvercf(m, "Solvercf");
    Solvercf
        .def("solve",
             (elsa::DataContainer<
                  std::complex<float>> & (elsa::Solver<std::complex<float>>::*) (long) )(
                 &elsa::Solver<std::complex<float>>::solve),
             py::arg("iterations") = static_cast<long>(0),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (elsa::DataContainer<std::complex<float>> & (elsa::Solver<std::complex<float>>::*) ())(
                 &elsa::Solver<std::complex<float>>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<std::complex<float>>& (
                 elsa::Solver<std::complex<float>>::*) ()
                  const)(&elsa::Solver<std::complex<float>>::getCurrentSolution),
             py::return_value_policy::reference_internal);

    py::class_<elsa::Cloneable<elsa::Solver<std::complex<double>>>> CloneableSolvercd(
        m, "CloneableSolvercd");
    CloneableSolvercd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Solver<std::complex<double>>>::*)(
                 const elsa::Solver<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Solver<std::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Solver<std::complex<double>>>::*)(
                 const elsa::Solver<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Solver<std::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Solver<std::complex<double>>,
                                       std::default_delete<elsa::Solver<std::complex<double>>>>(
                          elsa::Cloneable<elsa::Solver<std::complex<double>>>::*)()
                           const)(&elsa::Cloneable<elsa::Solver<std::complex<double>>>::clone));

    py::class_<elsa::Solver<std::complex<double>>,
               elsa::Cloneable<elsa::Solver<std::complex<double>>>>
        Solvercd(m, "Solvercd");
    Solvercd
        .def("solve",
             (elsa::DataContainer<
                  std::complex<double>> & (elsa::Solver<std::complex<double>>::*) (long) )(
                 &elsa::Solver<std::complex<double>>::solve),
             py::arg("iterations") = static_cast<long>(0),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (elsa::DataContainer<
                  std::complex<double>> & (elsa::Solver<std::complex<double>>::*) ())(
                 &elsa::Solver<std::complex<double>>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<std::complex<double>>& (
                 elsa::Solver<std::complex<double>>::*) ()
                  const)(&elsa::Solver<std::complex<double>>::getCurrentSolution),
             py::return_value_policy::reference_internal);

    py::class_<elsa::GradientDescent<float>, elsa::Solver<float>> GradientDescentf(
        m, "GradientDescentf");
    GradientDescentf
        .def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::GradientDescent<float>::*) ())(
                 &elsa::GradientDescent<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::GradientDescent<float>::*) ()
                  const)(&elsa::GradientDescent<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("stepSize"));

    m.attr("GradientDescent") = m.attr("GradientDescentf");

    py::class_<elsa::GradientDescent<double>, elsa::Solver<double>> GradientDescentd(
        m, "GradientDescentd");
    GradientDescentd
        .def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::GradientDescent<double>::*) ())(
                 &elsa::GradientDescent<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::GradientDescent<double>::*) ()
                  const)(&elsa::GradientDescent<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("stepSize"));

    py::class_<elsa::CG<float>, elsa::Solver<float>> CGf(m, "CGf");
    CGf.def("getCurrentSolution",
            (elsa::DataContainer<float> & (elsa::CG<float>::*) ())(
                &elsa::CG<float>::getCurrentSolution),
            py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::CG<float>::*) ()
                  const)(&elsa::CG<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&, float>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("CG") = m.attr("CGf");

    py::class_<elsa::CG<double>, elsa::Solver<double>> CGd(m, "CGd");
    CGd.def("getCurrentSolution",
            (elsa::DataContainer<double> & (elsa::CG<double>::*) ())(
                &elsa::CG<double>::getCurrentSolution),
            py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::CG<double>::*) ()
                  const)(&elsa::CG<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&, double>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    py::class_<elsa::ISTA<float>, elsa::Solver<float>> ISTAf(m, "ISTAf");
    ISTAf
        .def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::ISTA<float>::*) ())(
                 &elsa::ISTA<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::ISTA<float>::*) ()
                  const)(&elsa::ISTA<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, elsa::geometry::Threshold<float>>(),
             py::arg("problem"), py::arg("mu"))
        .def(py::init<const elsa::Problem<float>&, elsa::geometry::Threshold<float>, float>(),
             py::arg("problem"), py::arg("mu"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("ISTA") = m.attr("ISTAf");

    py::class_<elsa::ISTA<double>, elsa::Solver<double>> ISTAd(m, "ISTAd");
    ISTAd
        .def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::ISTA<double>::*) ())(
                 &elsa::ISTA<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::ISTA<double>::*) ()
                  const)(&elsa::ISTA<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, elsa::geometry::Threshold<double>>(),
             py::arg("problem"), py::arg("mu"))
        .def(py::init<const elsa::Problem<double>&, elsa::geometry::Threshold<double>, double>(),
             py::arg("problem"), py::arg("mu"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    py::class_<elsa::FISTA<float>, elsa::Solver<float>> FISTAf(m, "FISTAf");
    FISTAf
        .def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::FISTA<float>::*) ())(
                 &elsa::FISTA<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::FISTA<float>::*) ()
                  const)(&elsa::FISTA<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, elsa::geometry::Threshold<float>>(),
             py::arg("problem"), py::arg("mu"))
        .def(py::init<const elsa::Problem<float>&, elsa::geometry::Threshold<float>, float>(),
             py::arg("problem"), py::arg("mu"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("FISTA") = m.attr("FISTAf");

    py::class_<elsa::FISTA<double>, elsa::Solver<double>> FISTAd(m, "FISTAd");
    FISTAd
        .def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::FISTA<double>::*) ())(
                 &elsa::FISTA<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::FISTA<double>::*) ()
                  const)(&elsa::FISTA<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, elsa::geometry::Threshold<double>>(),
             py::arg("problem"), py::arg("mu"))
        .def(py::init<const elsa::Problem<double>&, elsa::geometry::Threshold<double>, double>(),
             py::arg("problem"), py::arg("mu"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    py::class_<elsa::FGM<float>, elsa::Solver<float>> FGMf(m, "FGMf");
    FGMf.def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::FGM<float>::*) ())(
                 &elsa::FGM<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::FGM<float>::*) ()
                  const)(&elsa::FGM<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&, float>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("FGM") = m.attr("FGMf");

    py::class_<elsa::FGM<double>, elsa::Solver<double>> FGMd(m, "FGMd");
    FGMd.def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::FGM<double>::*) ())(
                 &elsa::FGM<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::FGM<double>::*) ()
                  const)(&elsa::FGM<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&, double>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    py::class_<elsa::OGM<float>, elsa::Solver<float>> OGMf(m, "OGMf");
    OGMf.def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::OGM<float>::*) ())(
                 &elsa::OGM<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::OGM<float>::*) ()
                  const)(&elsa::OGM<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&, float>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("OGM") = m.attr("OGMf");

    py::class_<elsa::OGM<double>, elsa::Solver<double>> OGMd(m, "OGMd");
    OGMd.def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::OGM<double>::*) ())(
                 &elsa::OGM<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::OGM<double>::*) ()
                  const)(&elsa::OGM<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&>(),
             py::arg("problem"), py::arg("preconditionerInverse"))
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&, double>(),
             py::arg("problem"), py::arg("preconditionerInverse"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Problem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    py::class_<elsa::SQS<float>, elsa::Solver<float>> SQSf(m, "SQSf");
    SQSf.def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::SQS<float>::*) ())(
                 &elsa::SQS<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::SQS<float>::*) ()
                  const)(&elsa::SQS<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<float>&, bool>(), py::arg("problem"),
             py::arg("momentumAcceleration") = static_cast<bool>(true))
        .def(py::init<const elsa::Problem<float>&, bool, float>(), py::arg("problem"),
             py::arg("momentumAcceleration"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&, bool>(),
             py::arg("problem"), py::arg("preconditioner"),
             py::arg("momentumAcceleration") = static_cast<bool>(true))
        .def(py::init<const elsa::Problem<float>&, const elsa::LinearOperator<float>&, bool,
                      float>(),
             py::arg("problem"), py::arg("preconditioner"), py::arg("momentumAcceleration"),
             py::arg("epsilon"));

    m.attr("SQS") = m.attr("SQSf");

    py::class_<elsa::SQS<double>, elsa::Solver<double>> SQSd(m, "SQSd");
    SQSd.def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::SQS<double>::*) ())(
                 &elsa::SQS<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::SQS<double>::*) ()
                  const)(&elsa::SQS<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Problem<double>&, bool>(), py::arg("problem"),
             py::arg("momentumAcceleration") = static_cast<bool>(true))
        .def(py::init<const elsa::Problem<double>&, bool, double>(), py::arg("problem"),
             py::arg("momentumAcceleration"), py::arg("epsilon"))
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&, bool>(),
             py::arg("problem"), py::arg("preconditioner"),
             py::arg("momentumAcceleration") = static_cast<bool>(true))
        .def(py::init<const elsa::Problem<double>&, const elsa::LinearOperator<double>&, bool,
                      double>(),
             py::arg("problem"), py::arg("preconditioner"), py::arg("momentumAcceleration"),
             py::arg("epsilon"));

    py::class_<elsa::OrthogonalMatchingPursuit<float>, elsa::Solver<float>>
        OrthogonalMatchingPursuitf(m, "OrthogonalMatchingPursuitf");
    OrthogonalMatchingPursuitf
        .def("getCurrentSolution",
             (elsa::DataContainer<float> & (elsa::OrthogonalMatchingPursuit<float>::*) ())(
                 &elsa::OrthogonalMatchingPursuit<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<float>& (elsa::OrthogonalMatchingPursuit<float>::*) ()
                  const)(&elsa::OrthogonalMatchingPursuit<float>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::RepresentationProblem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::RepresentationProblem<float>&, float>(), py::arg("problem"),
             py::arg("epsilon"));

    m.attr("OrthogonalMatchingPursuit") = m.attr("OrthogonalMatchingPursuitf");

    py::class_<elsa::OrthogonalMatchingPursuit<double>, elsa::Solver<double>>
        OrthogonalMatchingPursuitd(m, "OrthogonalMatchingPursuitd");
    OrthogonalMatchingPursuitd
        .def("getCurrentSolution",
             (elsa::DataContainer<double> & (elsa::OrthogonalMatchingPursuit<double>::*) ())(
                 &elsa::OrthogonalMatchingPursuit<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def("getCurrentSolution",
             (const elsa::DataContainer<double>& (elsa::OrthogonalMatchingPursuit<double>::*) ()
                  const)(&elsa::OrthogonalMatchingPursuit<double>::getCurrentSolution),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::RepresentationProblem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::RepresentationProblem<double>&, double>(), py::arg("problem"),
             py::arg("epsilon"));

    elsa::SolversHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_solvers, m)
{
    add_definitions_pyelsa_solvers(m);
}
