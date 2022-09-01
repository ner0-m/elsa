#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include "JacobiPreconditioner.h"
#include "LASSOProblem.h"
#include "Problem.h"
#include "QuadricProblem.h"
#include "RegularizationTerm.h"
#include "RepresentationProblem.h"
#include "SplittingProblem.h"
#include "SubsetProblem.h"
#include "TikhonovProblem.h"
#include "WLSProblem.h"
#include "WLSSubsetProblem.h"

#include "hints/problems_hints.cpp"

namespace py = pybind11;

template <class data_t>
void add_problem_definitions(
    py::class_<elsa::Problem<data_t>, elsa::Cloneable<elsa::Problem<data_t>>> problem)
{
    problem
        .def("getGradient",
             (elsa::DataContainer<data_t>(elsa::Problem<data_t>::*)(
                 const elsa::DataContainer<data_t>&))(&elsa::Problem<data_t>::getGradient),
             py::return_value_policy::move)
        .def("getGradient",
             (void(elsa::Problem<data_t>::*)(const elsa::DataContainer<data_t>&,
                                             elsa::DataContainer<data_t>&))(
                 &elsa::Problem<data_t>::getGradient),
             py::arg("x"), py::arg("result"))
        .def("getHessian",
             (elsa::LinearOperator<data_t>(elsa::Problem<data_t>::*)(
                 const elsa::DataContainer<data_t>&) const)(&elsa::Problem<data_t>::getHessian),
             py::return_value_policy::move)
        .def("getDataTerm",
             (const elsa::Functional<data_t>& (elsa::Problem<data_t>::*) ()
                  const)(&elsa::Problem<data_t>::getDataTerm),
             py::return_value_policy::reference_internal)
        .def(
            "getRegularizationTerms",
            (const std::vector<elsa::RegularizationTerm<data_t>,
                               std::allocator<elsa::RegularizationTerm<data_t>>>& (
                elsa::Problem<data_t>::*) () const)(&elsa::Problem<data_t>::getRegularizationTerms),
            py::return_value_policy::reference_internal)
        .def("getLipschitzConstant",
             (data_t(elsa::Problem<data_t>::*)(const elsa::DataContainer<data_t>&, long)
                  const)(&elsa::Problem<data_t>::getLipschitzConstant),
             py::arg("x"), py::arg("nIterations") = static_cast<long>(5))
        .def("evaluate", (data_t(elsa::Problem<data_t>::*)(const elsa::DataContainer<data_t>&))(
                             &elsa::Problem<data_t>::evaluate))
        .def(py::init<const elsa::Functional<data_t>&>(), py::arg("dataTerm"))
        .def(py::init<const elsa::Functional<data_t>&, std::optional<data_t>>(),
             py::arg("dataTerm"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Functional<data_t>&, const elsa::RegularizationTerm<data_t>&>(),
             py::arg("dataTerm"), py::arg("regTerm"))
        .def(py::init<const elsa::Functional<data_t>&, const elsa::RegularizationTerm<data_t>&,
                      std::optional<data_t>>(),
             py::arg("dataTerm"), py::arg("regTerm"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Functional<data_t>&,
                      const std::vector<elsa::RegularizationTerm<data_t>,
                                        std::allocator<elsa::RegularizationTerm<data_t>>>&>(),
             py::arg("dataTerm"), py::arg("regTerms"))
        .def(py::init<const elsa::Functional<data_t>&,
                      const std::vector<elsa::RegularizationTerm<data_t>,
                                        std::allocator<elsa::RegularizationTerm<data_t>>>&,
                      std::optional<data_t>>(),
             py::arg("dataTerm"), py::arg("regTerms"), py::arg("lipschitzConstant"));
}

void add_definitions_pyelsa_problems(py::module& m)
{
    py::class_<elsa::RegularizationTerm<float>> RegularizationTermf(m, "RegularizationTermf");
    RegularizationTermf
        .def("__ne__",
             (bool(elsa::RegularizationTerm<float>::*)(const elsa::RegularizationTerm<float>&)
                  const)(&elsa::RegularizationTerm<float>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::RegularizationTerm<float>::*)(const elsa::RegularizationTerm<float>&)
                  const)(&elsa::RegularizationTerm<float>::operator==),
             py::arg("other"))
        .def("getFunctional",
             (elsa::Functional<float> & (elsa::RegularizationTerm<float>::*) () const)(
                 &elsa::RegularizationTerm<float>::getFunctional),
             py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::RegularizationTerm<
                 float> & (elsa::RegularizationTerm<float>::*) (const elsa::RegularizationTerm<float>&) )(
                &elsa::RegularizationTerm<float>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getWeight", (float(elsa::RegularizationTerm<float>::*)()
                               const)(&elsa::RegularizationTerm<float>::getWeight))
        .def(py::init<const elsa::RegularizationTerm<float>&>(), py::arg("other"))
        .def(py::init<float, const elsa::Functional<float>&>(), py::arg("weight"),
             py::arg("functional"));

    m.attr("RegularizationTerm") = m.attr("RegularizationTermf");

    py::class_<elsa::RegularizationTerm<double>> RegularizationTermd(m, "RegularizationTermd");
    RegularizationTermd
        .def("__ne__",
             (bool(elsa::RegularizationTerm<double>::*)(const elsa::RegularizationTerm<double>&)
                  const)(&elsa::RegularizationTerm<double>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::RegularizationTerm<double>::*)(const elsa::RegularizationTerm<double>&)
                  const)(&elsa::RegularizationTerm<double>::operator==),
             py::arg("other"))
        .def("getFunctional",
             (elsa::Functional<double> & (elsa::RegularizationTerm<double>::*) () const)(
                 &elsa::RegularizationTerm<double>::getFunctional),
             py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::RegularizationTerm<
                 double> & (elsa::RegularizationTerm<double>::*) (const elsa::RegularizationTerm<double>&) )(
                &elsa::RegularizationTerm<double>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getWeight", (double(elsa::RegularizationTerm<double>::*)()
                               const)(&elsa::RegularizationTerm<double>::getWeight))
        .def(py::init<const elsa::RegularizationTerm<double>&>(), py::arg("other"))
        .def(py::init<double, const elsa::Functional<double>&>(), py::arg("weight"),
             py::arg("functional"));

    py::class_<elsa::RegularizationTerm<std::complex<float>>> RegularizationTermcf(
        m, "RegularizationTermcf");
    RegularizationTermcf
        .def("__ne__",
             (bool(elsa::RegularizationTerm<std::complex<float>>::*)(
                 const elsa::RegularizationTerm<std::complex<float>>&)
                  const)(&elsa::RegularizationTerm<std::complex<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::RegularizationTerm<std::complex<float>>::*)(
                 const elsa::RegularizationTerm<std::complex<float>>&)
                  const)(&elsa::RegularizationTerm<std::complex<float>>::operator==),
             py::arg("other"))
        .def("getFunctional",
             (elsa::Functional<std::complex<
                  float>> & (elsa::RegularizationTerm<std::complex<float>>::*) () const)(
                 &elsa::RegularizationTerm<std::complex<float>>::getFunctional),
             py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::RegularizationTerm<std::complex<
                 float>> & (elsa::RegularizationTerm<std::complex<float>>::*) (const elsa::RegularizationTerm<std::complex<float>>&) )(
                &elsa::RegularizationTerm<std::complex<float>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getWeight",
             (std::complex<float>(elsa::RegularizationTerm<std::complex<float>>::*)()
                  const)(&elsa::RegularizationTerm<std::complex<float>>::getWeight),
             py::return_value_policy::move)
        .def(py::init<std::complex<float>, const elsa::Functional<std::complex<float>>&>(),
             py::arg("weight"), py::arg("functional"))
        .def(py::init<const elsa::RegularizationTerm<std::complex<float>>&>(), py::arg("other"));

    py::class_<elsa::RegularizationTerm<std::complex<double>>> RegularizationTermcd(
        m, "RegularizationTermcd");
    RegularizationTermcd
        .def("__ne__",
             (bool(elsa::RegularizationTerm<std::complex<double>>::*)(
                 const elsa::RegularizationTerm<std::complex<double>>&)
                  const)(&elsa::RegularizationTerm<std::complex<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::RegularizationTerm<std::complex<double>>::*)(
                 const elsa::RegularizationTerm<std::complex<double>>&)
                  const)(&elsa::RegularizationTerm<std::complex<double>>::operator==),
             py::arg("other"))
        .def("getFunctional",
             (elsa::Functional<std::complex<
                  double>> & (elsa::RegularizationTerm<std::complex<double>>::*) () const)(
                 &elsa::RegularizationTerm<std::complex<double>>::getFunctional),
             py::return_value_policy::reference_internal)
        .def(
            "set",
            (elsa::RegularizationTerm<std::complex<
                 double>> & (elsa::RegularizationTerm<std::complex<double>>::*) (const elsa::RegularizationTerm<std::complex<double>>&) )(
                &elsa::RegularizationTerm<std::complex<double>>::operator=),
            py::arg("other"), py::return_value_policy::reference_internal)
        .def("getWeight",
             (std::complex<double>(elsa::RegularizationTerm<std::complex<double>>::*)()
                  const)(&elsa::RegularizationTerm<std::complex<double>>::getWeight),
             py::return_value_policy::move)
        .def(py::init<std::complex<double>, const elsa::Functional<std::complex<double>>&>(),
             py::arg("weight"), py::arg("functional"))
        .def(py::init<const elsa::RegularizationTerm<std::complex<double>>&>(), py::arg("other"));

    py::class_<elsa::Cloneable<elsa::Problem<float>>> CloneableProblemf(m, "CloneableProblemf");
    CloneableProblemf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Problem<float>>::*)(const elsa::Problem<float>&)
                  const)(&elsa::Cloneable<elsa::Problem<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Problem<float>>::*)(const elsa::Problem<float>&)
                  const)(&elsa::Cloneable<elsa::Problem<float>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Problem<float>, std::default_delete<elsa::Problem<float>>>(
                 elsa::Cloneable<elsa::Problem<float>>::*)()
                  const)(&elsa::Cloneable<elsa::Problem<float>>::clone));

    py::class_<elsa::Problem<float>, elsa::Cloneable<elsa::Problem<float>>> Problemf(m, "Problemf");
    add_problem_definitions<float>(Problemf);
    m.attr("Problem") = m.attr("Problemf");

    py::class_<elsa::Cloneable<elsa::Problem<double>>> CloneableProblemd(m, "CloneableProblemd");
    CloneableProblemd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Problem<double>>::*)(const elsa::Problem<double>&)
                  const)(&elsa::Cloneable<elsa::Problem<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Problem<double>>::*)(const elsa::Problem<double>&)
                  const)(&elsa::Cloneable<elsa::Problem<double>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Problem<double>, std::default_delete<elsa::Problem<double>>>(
                 elsa::Cloneable<elsa::Problem<double>>::*)()
                  const)(&elsa::Cloneable<elsa::Problem<double>>::clone));

    py::class_<elsa::Problem<double>, elsa::Cloneable<elsa::Problem<double>>> Problemd(m,
                                                                                       "Problemd");
    add_problem_definitions<double>(Problemd);

    py::class_<elsa::Cloneable<elsa::Problem<std::complex<float>>>> CloneableProblemcf(
        m, "CloneableProblemcf");
    CloneableProblemcf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Problem<std::complex<float>>>::*)(
                 const elsa::Problem<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Problem<std::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Problem<std::complex<float>>>::*)(
                 const elsa::Problem<std::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Problem<std::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Problem<std::complex<float>>,
                                       std::default_delete<elsa::Problem<std::complex<float>>>>(
                          elsa::Cloneable<elsa::Problem<std::complex<float>>>::*)()
                           const)(&elsa::Cloneable<elsa::Problem<std::complex<float>>>::clone));

    py::class_<elsa::Problem<std::complex<float>>,
               elsa::Cloneable<elsa::Problem<std::complex<float>>>>
        Problemcf(m, "Problemcf");
    add_problem_definitions<std::complex<float>>(Problemcf);

    py::class_<elsa::Cloneable<elsa::Problem<std::complex<double>>>> CloneableProblemcd(
        m, "CloneableProblemcd");
    CloneableProblemcd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Problem<std::complex<double>>>::*)(
                 const elsa::Problem<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Problem<std::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Problem<std::complex<double>>>::*)(
                 const elsa::Problem<std::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Problem<std::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Problem<std::complex<double>>,
                                       std::default_delete<elsa::Problem<std::complex<double>>>>(
                          elsa::Cloneable<elsa::Problem<std::complex<double>>>::*)()
                           const)(&elsa::Cloneable<elsa::Problem<std::complex<double>>>::clone));

    py::class_<elsa::Problem<std::complex<double>>,
               elsa::Cloneable<elsa::Problem<std::complex<double>>>>
        Problemcd(m, "Problemcd");
    add_problem_definitions<std::complex<double>>(Problemcd);

    py::class_<elsa::WLSProblem<float>, elsa::Problem<float>> WLSProblemf(m, "WLSProblemf");
    WLSProblemf
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&,
                      std::optional<float>>(),
             py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Scaling<float>&, const elsa::LinearOperator<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("W"), py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Scaling<float>&, const elsa::LinearOperator<float>&,
                      const elsa::DataContainer<float>&, std::optional<float>>(),
             py::arg("W"), py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::WLSProblem<float>&>());

    m.attr("WLSProblem") = m.attr("WLSProblemf");

    py::class_<elsa::WLSProblem<double>, elsa::Problem<double>> WLSProblemd(m, "WLSProblemd");
    WLSProblemd
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&,
                      std::optional<double>>(),
             py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Scaling<double>&, const elsa::LinearOperator<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("W"), py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Scaling<double>&, const elsa::LinearOperator<double>&,
                      const elsa::DataContainer<double>&, std::optional<double>>(),
             py::arg("W"), py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::WLSProblem<double>&>());

    py::class_<elsa::WLSProblem<std::complex<float>>, elsa::Problem<std::complex<float>>>
        WLSProblemcf(m, "WLSProblemcf");
    WLSProblemcf
        .def(py::init<const elsa::LinearOperator<std::complex<float>>&,
                      const elsa::DataContainer<std::complex<float>>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::LinearOperator<std::complex<float>>&,
                      const elsa::DataContainer<std::complex<float>>&,
                      std::optional<std::complex<float>>>(),
             py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Problem<std::complex<float>>&>(), py::arg("problem"))
        .def(py::init<const elsa::Scaling<std::complex<float>>&,
                      const elsa::LinearOperator<std::complex<float>>&,
                      const elsa::DataContainer<std::complex<float>>&>(),
             py::arg("W"), py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Scaling<std::complex<float>>&,
                      const elsa::LinearOperator<std::complex<float>>&,
                      const elsa::DataContainer<std::complex<float>>&,
                      std::optional<std::complex<float>>>(),
             py::arg("W"), py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::WLSProblem<std::complex<float>>&>());

    py::class_<elsa::WLSProblem<std::complex<double>>, elsa::Problem<std::complex<double>>>
        WLSProblemcd(m, "WLSProblemcd");
    WLSProblemcd
        .def(py::init<const elsa::LinearOperator<std::complex<double>>&,
                      const elsa::DataContainer<std::complex<double>>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::LinearOperator<std::complex<double>>&,
                      const elsa::DataContainer<std::complex<double>>&,
                      std::optional<std::complex<double>>>(),
             py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::Problem<std::complex<double>>&>(), py::arg("problem"))
        .def(py::init<const elsa::Scaling<std::complex<double>>&,
                      const elsa::LinearOperator<std::complex<double>>&,
                      const elsa::DataContainer<std::complex<double>>&>(),
             py::arg("W"), py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Scaling<std::complex<double>>&,
                      const elsa::LinearOperator<std::complex<double>>&,
                      const elsa::DataContainer<std::complex<double>>&,
                      std::optional<std::complex<double>>>(),
             py::arg("W"), py::arg("A"), py::arg("b"), py::arg("lipschitzConstant"))
        .def(py::init<const elsa::WLSProblem<std::complex<double>>&>());

    py::class_<elsa::TikhonovProblem<float>, elsa::Problem<float>> TikhonovProblemf(
        m, "TikhonovProblemf");
    TikhonovProblemf
        .def(
            py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>, float>(),
            py::arg("A"), py::arg("b"), py::arg("weight") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::WLSProblem<float>&, const elsa::RegularizationTerm<float>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::WLSProblem<float>&,
                      const std::vector<elsa::RegularizationTerm<float>,
                                        std::allocator<elsa::RegularizationTerm<float>>>&>(),
             py::arg("wlsProblem"), py::arg("regTerms"));

    m.attr("TikhonovProblem") = m.attr("TikhonovProblemf");

    py::class_<elsa::TikhonovProblem<double>, elsa::Problem<double>> TikhonovProblemd(
        m, "TikhonovProblemd");
    TikhonovProblemd
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>,
                      float>(),
             py::arg("A"), py::arg("b"), py::arg("weight") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::WLSProblem<double>&, const elsa::RegularizationTerm<double>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::WLSProblem<double>&,
                      const std::vector<elsa::RegularizationTerm<double>,
                                        std::allocator<elsa::RegularizationTerm<double>>>&>(),
             py::arg("wlsProblem"), py::arg("regTerms"));

    py::class_<elsa::TikhonovProblem<std::complex<float>>, elsa::Problem<std::complex<float>>>
        TikhonovProblemcf(m, "TikhonovProblemcf");
    TikhonovProblemcf
        .def(py::init<const elsa::LinearOperator<std::complex<float>>&,
                      const elsa::DataContainer<std::complex<float>>, float>(),
             py::arg("A"), py::arg("b"), py::arg("weight") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::WLSProblem<std::complex<float>>&,
                      const elsa::RegularizationTerm<std::complex<float>>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::WLSProblem<std::complex<float>>&,
                      const std::vector<
                          elsa::RegularizationTerm<std::complex<float>>,
                          std::allocator<elsa::RegularizationTerm<std::complex<float>>>>&>(),
             py::arg("wlsProblem"), py::arg("regTerms"));

    py::class_<elsa::TikhonovProblem<std::complex<double>>, elsa::Problem<std::complex<double>>>
        TikhonovProblemcd(m, "TikhonovProblemcd");
    TikhonovProblemcd
        .def(py::init<const elsa::LinearOperator<std::complex<double>>&,
                      const elsa::DataContainer<std::complex<double>>, float>(),
             py::arg("A"), py::arg("b"), py::arg("weight") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::WLSProblem<std::complex<double>>&,
                      const elsa::RegularizationTerm<std::complex<double>>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::WLSProblem<std::complex<double>>&,
                      const std::vector<
                          elsa::RegularizationTerm<std::complex<double>>,
                          std::allocator<elsa::RegularizationTerm<std::complex<double>>>>&>(),
             py::arg("wlsProblem"), py::arg("regTerms"));

    py::class_<elsa::QuadricProblem<float>, elsa::Problem<float>> QuadricProblemf(
        m, "QuadricProblemf");
    QuadricProblemf
        .def(
            py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&, bool>(),
            py::arg("A"), py::arg("b"), py::arg("spdA"))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"))
        .def(py::init<const elsa::Quadric<float>&>(), py::arg("quadric"))
        .def(py::init<const elsa::QuadricProblem<float>&>());

    m.attr("QuadricProblem") = m.attr("QuadricProblemf");

    py::class_<elsa::QuadricProblem<double>, elsa::Problem<double>> QuadricProblemd(
        m, "QuadricProblemd");
    QuadricProblemd
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&,
                      bool>(),
             py::arg("A"), py::arg("b"), py::arg("spdA"))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"))
        .def(py::init<const elsa::Quadric<double>&>(), py::arg("quadric"))
        .def(py::init<const elsa::QuadricProblem<double>&>());

    py::class_<elsa::LASSOProblem<float>, elsa::Problem<float>> LASSOProblemf(m, "LASSOProblemf");
    LASSOProblemf
        .def(py::init<elsa::WLSProblem<float>, const elsa::RegularizationTerm<float>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::LASSOProblem<float>&>())
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&,
                      float>(),
             py::arg("A"), py::arg("b"), py::arg("lambda") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::Problem<float>&>(), py::arg("problem"));

    m.attr("LASSOProblem") = m.attr("LASSOProblemf");

    py::class_<elsa::LASSOProblem<double>, elsa::Problem<double>> LASSOProblemd(m, "LASSOProblemd");
    LASSOProblemd
        .def(py::init<elsa::WLSProblem<double>, const elsa::RegularizationTerm<double>&>(),
             py::arg("wlsProblem"), py::arg("regTerm"))
        .def(py::init<const elsa::LASSOProblem<double>&>())
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&,
                      float>(),
             py::arg("A"), py::arg("b"), py::arg("lambda") = static_cast<float>(5.000000e-01))
        .def(py::init<const elsa::Problem<double>&>(), py::arg("problem"));

    py::class_<elsa::SplittingProblem<float>, elsa::Problem<float>> SplittingProblemf(
        m, "SplittingProblemf");
    SplittingProblemf
        .def("getConstraint",
             (const elsa::Constraint<float>& (elsa::SplittingProblem<float>::*) ()
                  const)(&elsa::SplittingProblem<float>::getConstraint),
             py::return_value_policy::reference_internal)
        .def("getF",
             (const elsa::Functional<float>& (elsa::SplittingProblem<float>::*) ()
                  const)(&elsa::SplittingProblem<float>::getF),
             py::return_value_policy::reference_internal)
        .def("getG",
             (const std::vector<elsa::RegularizationTerm<float>,
                                std::allocator<elsa::RegularizationTerm<float>>>& (
                 elsa::SplittingProblem<float>::*) () const)(&elsa::SplittingProblem<float>::getG),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Functional<float>&, const elsa::RegularizationTerm<float>&,
                      const elsa::Constraint<float>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"))
        .def(py::init<const elsa::Functional<float>&,
                      const std::vector<elsa::RegularizationTerm<float>,
                                        std::allocator<elsa::RegularizationTerm<float>>>&,
                      const elsa::Constraint<float>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"));

    m.attr("SplittingProblem") = m.attr("SplittingProblemf");

    py::class_<elsa::SplittingProblem<std::complex<float>>, elsa::Problem<std::complex<float>>>
        SplittingProblemcf(m, "SplittingProblemcf");
    SplittingProblemcf
        .def("getConstraint",
             (const elsa::Constraint<std::complex<float>>& (
                 elsa::SplittingProblem<std::complex<float>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<float>>::getConstraint),
             py::return_value_policy::reference_internal)
        .def("getF",
             (const elsa::Functional<std::complex<float>>& (
                 elsa::SplittingProblem<std::complex<float>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<float>>::getF),
             py::return_value_policy::reference_internal)
        .def("getG",
             (const std::vector<elsa::RegularizationTerm<std::complex<float>>,
                                std::allocator<elsa::RegularizationTerm<std::complex<float>>>>& (
                 elsa::SplittingProblem<std::complex<float>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<float>>::getG),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Functional<std::complex<float>>&,
                      const elsa::RegularizationTerm<std::complex<float>>&,
                      const elsa::Constraint<std::complex<float>>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"))
        .def(py::init<
                 const elsa::Functional<std::complex<float>>&,
                 const std::vector<elsa::RegularizationTerm<std::complex<float>>,
                                   std::allocator<elsa::RegularizationTerm<std::complex<float>>>>&,
                 const elsa::Constraint<std::complex<float>>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"));

    py::class_<elsa::SplittingProblem<double>, elsa::Problem<double>> SplittingProblemd(
        m, "SplittingProblemd");
    SplittingProblemd
        .def("getConstraint",
             (const elsa::Constraint<double>& (elsa::SplittingProblem<double>::*) ()
                  const)(&elsa::SplittingProblem<double>::getConstraint),
             py::return_value_policy::reference_internal)
        .def("getF",
             (const elsa::Functional<double>& (elsa::SplittingProblem<double>::*) ()
                  const)(&elsa::SplittingProblem<double>::getF),
             py::return_value_policy::reference_internal)
        .def(
            "getG",
            (const std::vector<elsa::RegularizationTerm<double>,
                               std::allocator<elsa::RegularizationTerm<double>>>& (
                elsa::SplittingProblem<double>::*) () const)(&elsa::SplittingProblem<double>::getG),
            py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Functional<double>&, const elsa::RegularizationTerm<double>&,
                      const elsa::Constraint<double>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"))
        .def(py::init<const elsa::Functional<double>&,
                      const std::vector<elsa::RegularizationTerm<double>,
                                        std::allocator<elsa::RegularizationTerm<double>>>&,
                      const elsa::Constraint<double>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"));

    py::class_<elsa::SplittingProblem<std::complex<double>>, elsa::Problem<std::complex<double>>>
        SplittingProblemcd(m, "SplittingProblemcd");
    SplittingProblemcd
        .def("getConstraint",
             (const elsa::Constraint<std::complex<double>>& (
                 elsa::SplittingProblem<std::complex<double>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<double>>::getConstraint),
             py::return_value_policy::reference_internal)
        .def("getF",
             (const elsa::Functional<std::complex<double>>& (
                 elsa::SplittingProblem<std::complex<double>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<double>>::getF),
             py::return_value_policy::reference_internal)
        .def("getG",
             (const std::vector<elsa::RegularizationTerm<std::complex<double>>,
                                std::allocator<elsa::RegularizationTerm<std::complex<double>>>>& (
                 elsa::SplittingProblem<std::complex<double>>::*) ()
                  const)(&elsa::SplittingProblem<std::complex<double>>::getG),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Functional<std::complex<double>>&,
                      const elsa::RegularizationTerm<std::complex<double>>&,
                      const elsa::Constraint<std::complex<double>>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"))
        .def(py::init<
                 const elsa::Functional<std::complex<double>>&,
                 const std::vector<elsa::RegularizationTerm<std::complex<double>>,
                                   std::allocator<elsa::RegularizationTerm<std::complex<double>>>>&,
                 const elsa::Constraint<std::complex<double>>&>(),
             py::arg("f"), py::arg("g"), py::arg("constraint"));

    py::class_<elsa::SubsetProblem<float>, elsa::Problem<float>> SubsetProblemf(m,
                                                                                "SubsetProblemf");
    SubsetProblemf
        .def("getSubsetGradient",
             (elsa::DataContainer<float>(elsa::SubsetProblem<float>::*)(
                 const elsa::DataContainer<float>&, long))(
                 &elsa::SubsetProblem<float>::getSubsetGradient),
             py::arg("x"), py::arg("subset"), py::return_value_policy::move)
        .def("getNumberOfSubsets", (long(elsa::SubsetProblem<float>::*)()
                                        const)(&elsa::SubsetProblem<float>::getNumberOfSubsets))
        .def("getSubsetGradient",
             (void(elsa::SubsetProblem<float>::*)(const elsa::DataContainer<float>&,
                                                  elsa::DataContainer<float>&, long))(
                 &elsa::SubsetProblem<float>::getSubsetGradient),
             py::arg("x"), py::arg("result"), py::arg("subset"));

    m.attr("SubsetProblem") = m.attr("SubsetProblemf");

    py::class_<elsa::SubsetProblem<double>, elsa::Problem<double>> SubsetProblemd(m,
                                                                                  "SubsetProblemd");
    SubsetProblemd
        .def("getSubsetGradient",
             (elsa::DataContainer<double>(elsa::SubsetProblem<double>::*)(
                 const elsa::DataContainer<double>&, long))(
                 &elsa::SubsetProblem<double>::getSubsetGradient),
             py::arg("x"), py::arg("subset"), py::return_value_policy::move)
        .def("getNumberOfSubsets", (long(elsa::SubsetProblem<double>::*)()
                                        const)(&elsa::SubsetProblem<double>::getNumberOfSubsets))
        .def("getSubsetGradient",
             (void(elsa::SubsetProblem<double>::*)(const elsa::DataContainer<double>&,
                                                   elsa::DataContainer<double>&, long))(
                 &elsa::SubsetProblem<double>::getSubsetGradient),
             py::arg("x"), py::arg("result"), py::arg("subset"));

    py::class_<elsa::WLSSubsetProblem<float>, elsa::SubsetProblem<float>> WLSSubsetProblemf(
        m, "WLSSubsetProblemf");
    elsa::WLSSubsetProblemHints<float>::addCustomMethods(WLSSubsetProblemf);

    m.attr("WLSSubsetProblem") = m.attr("WLSSubsetProblemf");

    py::class_<elsa::WLSSubsetProblem<double>, elsa::SubsetProblem<double>> WLSSubsetProblemd(
        m, "WLSSubsetProblemd");
    elsa::WLSSubsetProblemHints<double>::addCustomMethods(WLSSubsetProblemd);

    py::class_<elsa::JacobiPreconditioner<float>, elsa::LinearOperator<float>>
        JacobiPreconditionerf(m, "JacobiPreconditionerf");
    JacobiPreconditionerf.def(py::init<const elsa::LinearOperator<float>&, bool>(), py::arg("op"),
                              py::arg("inverse"));

    m.attr("JacobiPreconditioner") = m.attr("JacobiPreconditionerf");

    py::class_<elsa::JacobiPreconditioner<double>, elsa::LinearOperator<double>>
        JacobiPreconditionerd(m, "JacobiPreconditionerd");
    JacobiPreconditionerd.def(py::init<const elsa::LinearOperator<double>&, bool>(), py::arg("op"),
                              py::arg("inverse"));

    py::class_<elsa::RepresentationProblem<float>, elsa::Problem<float>> RepresentationProblemf(
        m, "RepresentationProblemf");
    RepresentationProblemf
        .def("getSignal",
             (const elsa::DataContainer<float>& (elsa::RepresentationProblem<float>::*) ()
                  const)(&elsa::RepresentationProblem<float>::getSignal),
             py::return_value_policy::reference_internal)
        .def("getDictionary",
             (const elsa::Dictionary<float>& (elsa::RepresentationProblem<float>::*) ()
                  const)(&elsa::RepresentationProblem<float>::getDictionary),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Dictionary<float>&, const elsa::DataContainer<float>&>(),
             py::arg("D"), py::arg("y"));

    m.attr("RepresentationProblem") = m.attr("RepresentationProblemf");

    py::class_<elsa::RepresentationProblem<double>, elsa::Problem<double>> RepresentationProblemd(
        m, "RepresentationProblemd");
    RepresentationProblemd
        .def("getSignal",
             (const elsa::DataContainer<double>& (elsa::RepresentationProblem<double>::*) ()
                  const)(&elsa::RepresentationProblem<double>::getSignal),
             py::return_value_policy::reference_internal)
        .def("getDictionary",
             (const elsa::Dictionary<double>& (elsa::RepresentationProblem<double>::*) ()
                  const)(&elsa::RepresentationProblem<double>::getDictionary),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Dictionary<double>&, const elsa::DataContainer<double>&>(),
             py::arg("D"), py::arg("y"));

    elsa::ProblemsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_problems, m)
{
    add_definitions_pyelsa_problems(m);
}
