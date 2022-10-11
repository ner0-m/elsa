#include <pybind11/pybind11.h>
#include <pybind11/complex.h>

#include "Constraint.h"
#include "EmissionLogLikelihood.h"
#include "Functional.h"
#include "Huber.h"
#include "L0PseudoNorm.h"
#include "L1Norm.h"
#include "L2NormPow2.h"
#include "LInfNorm.h"
#include "LinearResidual.h"
#include "PseudoHuber.h"
#include "Quadric.h"
#include "Residual.h"
#include "TransmissionLogLikelihood.h"
#include "WeightedL1Norm.h"
#include "WeightedL2NormPow2.h"

#include "hints/functionals_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_functionals(py::module& m)
{
    py::class_<elsa::Cloneable<elsa::Residual<float>>> CloneableResidualf(m, "CloneableResidualf");
    CloneableResidualf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Residual<float>>::*)(const elsa::Residual<float>&)
                  const)(&elsa::Cloneable<elsa::Residual<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Residual<float>>::*)(const elsa::Residual<float>&)
                  const)(&elsa::Cloneable<elsa::Residual<float>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Residual<float>, std::default_delete<elsa::Residual<float>>>(
                 elsa::Cloneable<elsa::Residual<float>>::*)()
                  const)(&elsa::Cloneable<elsa::Residual<float>>::clone));

    py::class_<elsa::Residual<float>, elsa::Cloneable<elsa::Residual<float>>> Residualf(
        m, "Residualf");
    Residualf
        .def("evaluate",
             (elsa::DataContainer<float>(elsa::Residual<float>::*)(
                 const elsa::DataContainer<float>&) const)(&elsa::Residual<float>::evaluate),
             py::arg("x"), py::return_value_policy::move)
        .def("getJacobian",
             (elsa::LinearOperator<float>(elsa::Residual<float>::*)(
                 const elsa::DataContainer<float>&))(&elsa::Residual<float>::getJacobian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<float>::*) ()
                  const)(&elsa::Residual<float>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<float>::*) ()
                  const)(&elsa::Residual<float>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (void(elsa::Residual<float>::*)(const elsa::DataContainer<float>&,
                                             elsa::DataContainer<float>&)
                  const)(&elsa::Residual<float>::evaluate),
             py::arg("x"), py::arg("result"));

    m.attr("Residual") = m.attr("Residualf");

    py::class_<elsa::Cloneable<elsa::Residual<double>>> CloneableResiduald(m, "CloneableResiduald");
    CloneableResiduald
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Residual<double>>::*)(const elsa::Residual<double>&)
                  const)(&elsa::Cloneable<elsa::Residual<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Residual<double>>::*)(const elsa::Residual<double>&)
                  const)(&elsa::Cloneable<elsa::Residual<double>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Residual<double>, std::default_delete<elsa::Residual<double>>>(
                 elsa::Cloneable<elsa::Residual<double>>::*)()
                  const)(&elsa::Cloneable<elsa::Residual<double>>::clone));

    py::class_<elsa::Residual<double>, elsa::Cloneable<elsa::Residual<double>>> Residuald(
        m, "Residuald");
    Residuald
        .def("evaluate",
             (elsa::DataContainer<double>(elsa::Residual<double>::*)(
                 const elsa::DataContainer<double>&) const)(&elsa::Residual<double>::evaluate),
             py::arg("x"), py::return_value_policy::move)
        .def("getJacobian",
             (elsa::LinearOperator<double>(elsa::Residual<double>::*)(
                 const elsa::DataContainer<double>&))(&elsa::Residual<double>::getJacobian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<double>::*) ()
                  const)(&elsa::Residual<double>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<double>::*) ()
                  const)(&elsa::Residual<double>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (void(elsa::Residual<double>::*)(const elsa::DataContainer<double>&,
                                              elsa::DataContainer<double>&)
                  const)(&elsa::Residual<double>::evaluate),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::Cloneable<elsa::Residual<thrust::complex<float>>>> CloneableResidualcf(
        m, "CloneableResidualcf");
    CloneableResidualcf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::*)(
                 const elsa::Residual<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::*)(
                 const elsa::Residual<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Residual<thrust::complex<float>>,
                                       std::default_delete<elsa::Residual<thrust::complex<float>>>>(
                          elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::*)()
                           const)(&elsa::Cloneable<elsa::Residual<thrust::complex<float>>>::clone));

    py::class_<elsa::Residual<thrust::complex<float>>,
               elsa::Cloneable<elsa::Residual<thrust::complex<float>>>>
        Residualcf(m, "Residualcf");
    Residualcf
        .def(
            "evaluate",
            (elsa::DataContainer<thrust::complex<float>>(elsa::Residual<thrust::complex<float>>::*)(
                const elsa::DataContainer<thrust::complex<float>>&)
                 const)(&elsa::Residual<thrust::complex<float>>::evaluate),
            py::arg("x"), py::return_value_policy::move)
        .def("getJacobian",
             (elsa::LinearOperator<thrust::complex<float>>(
                 elsa::Residual<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&))(
                 &elsa::Residual<thrust::complex<float>>::getJacobian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<thrust::complex<float>>::*) ()
                  const)(&elsa::Residual<thrust::complex<float>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<thrust::complex<float>>::*) ()
                  const)(&elsa::Residual<thrust::complex<float>>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (void(elsa::Residual<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&,
                 elsa::DataContainer<thrust::complex<float>>&)
                  const)(&elsa::Residual<thrust::complex<float>>::evaluate),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::Cloneable<elsa::Residual<thrust::complex<double>>>> CloneableResidualcd(
        m, "CloneableResidualcd");
    CloneableResidualcd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::*)(
                 const elsa::Residual<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::*)(
                 const elsa::Residual<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Residual<thrust::complex<double>>,
                              std::default_delete<elsa::Residual<thrust::complex<double>>>>(
                 elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::*)()
                  const)(&elsa::Cloneable<elsa::Residual<thrust::complex<double>>>::clone));

    py::class_<elsa::Residual<thrust::complex<double>>,
               elsa::Cloneable<elsa::Residual<thrust::complex<double>>>>
        Residualcd(m, "Residualcd");
    Residualcd
        .def("evaluate",
             (elsa::DataContainer<thrust::complex<double>>(
                 elsa::Residual<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&)
                  const)(&elsa::Residual<thrust::complex<double>>::evaluate),
             py::arg("x"), py::return_value_policy::move)
        .def("getJacobian",
             (elsa::LinearOperator<thrust::complex<double>>(
                 elsa::Residual<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&))(
                 &elsa::Residual<thrust::complex<double>>::getJacobian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<thrust::complex<double>>::*) ()
                  const)(&elsa::Residual<thrust::complex<double>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::Residual<thrust::complex<double>>::*) ()
                  const)(&elsa::Residual<thrust::complex<double>>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (void(elsa::Residual<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&,
                 elsa::DataContainer<thrust::complex<double>>&)
                  const)(&elsa::Residual<thrust::complex<double>>::evaluate),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::LinearResidual<float>, elsa::Residual<float>> LinearResidualf(
        m, "LinearResidualf");
    LinearResidualf
        .def("hasDataVector", (bool(elsa::LinearResidual<float>::*)()
                                   const)(&elsa::LinearResidual<float>::hasDataVector))
        .def("hasOperator", (bool(elsa::LinearResidual<float>::*)()
                                 const)(&elsa::LinearResidual<float>::hasOperator))
        .def("getDataVector",
             (const elsa::DataContainer<float>& (elsa::LinearResidual<float>::*) ()
                  const)(&elsa::LinearResidual<float>::getDataVector),
             py::return_value_policy::reference_internal)
        .def("getOperator",
             (const elsa::LinearOperator<float>& (elsa::LinearResidual<float>::*) ()
                  const)(&elsa::LinearResidual<float>::getOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<float>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"))
        .def(py::init<const elsa::LinearOperator<float>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
             py::arg("A"), py::arg("b"));

    m.attr("LinearResidual") = m.attr("LinearResidualf");

    py::class_<elsa::LinearResidual<double>, elsa::Residual<double>> LinearResiduald(
        m, "LinearResiduald");
    LinearResiduald
        .def("hasDataVector", (bool(elsa::LinearResidual<double>::*)()
                                   const)(&elsa::LinearResidual<double>::hasDataVector))
        .def("hasOperator", (bool(elsa::LinearResidual<double>::*)()
                                 const)(&elsa::LinearResidual<double>::hasOperator))
        .def("getDataVector",
             (const elsa::DataContainer<double>& (elsa::LinearResidual<double>::*) ()
                  const)(&elsa::LinearResidual<double>::getDataVector),
             py::return_value_policy::reference_internal)
        .def("getOperator",
             (const elsa::LinearOperator<double>& (elsa::LinearResidual<double>::*) ()
                  const)(&elsa::LinearResidual<double>::getOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<double>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"))
        .def(py::init<const elsa::LinearOperator<double>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::LinearResidual<thrust::complex<float>>, elsa::Residual<thrust::complex<float>>>
        LinearResidualcf(m, "LinearResidualcf");
    LinearResidualcf
        .def("hasDataVector", (bool(elsa::LinearResidual<thrust::complex<float>>::*)() const)(
                                  &elsa::LinearResidual<thrust::complex<float>>::hasDataVector))
        .def("hasOperator", (bool(elsa::LinearResidual<thrust::complex<float>>::*)()
                                 const)(&elsa::LinearResidual<thrust::complex<float>>::hasOperator))
        .def("getDataVector",
             (const elsa::DataContainer<thrust::complex<float>>& (
                 elsa::LinearResidual<thrust::complex<float>>::*) ()
                  const)(&elsa::LinearResidual<thrust::complex<float>>::getDataVector),
             py::return_value_policy::reference_internal)
        .def("getOperator",
             (const elsa::LinearOperator<thrust::complex<float>>& (
                 elsa::LinearResidual<thrust::complex<float>>::*) ()
                  const)(&elsa::LinearResidual<thrust::complex<float>>::getOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<thrust::complex<float>>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&,
                      const elsa::DataContainer<thrust::complex<float>>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::LinearResidual<thrust::complex<double>>,
               elsa::Residual<thrust::complex<double>>>
        LinearResidualcd(m, "LinearResidualcd");
    LinearResidualcd
        .def("hasDataVector", (bool(elsa::LinearResidual<thrust::complex<double>>::*)() const)(
                                  &elsa::LinearResidual<thrust::complex<double>>::hasDataVector))
        .def("hasOperator", (bool(elsa::LinearResidual<thrust::complex<double>>::*)() const)(
                                &elsa::LinearResidual<thrust::complex<double>>::hasOperator))
        .def("getDataVector",
             (const elsa::DataContainer<thrust::complex<double>>& (
                 elsa::LinearResidual<thrust::complex<double>>::*) ()
                  const)(&elsa::LinearResidual<thrust::complex<double>>::getDataVector),
             py::return_value_policy::reference_internal)
        .def("getOperator",
             (const elsa::LinearOperator<thrust::complex<double>>& (
                 elsa::LinearResidual<thrust::complex<double>>::*) ()
                  const)(&elsa::LinearResidual<thrust::complex<double>>::getOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<thrust::complex<double>>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&,
                      const elsa::DataContainer<thrust::complex<double>>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::Cloneable<elsa::Functional<float>>> CloneableFunctionalf(
        m, "CloneableFunctionalf");
    CloneableFunctionalf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Functional<float>>::*)(const elsa::Functional<float>&)
                  const)(&elsa::Cloneable<elsa::Functional<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Functional<float>>::*)(const elsa::Functional<float>&)
                  const)(&elsa::Cloneable<elsa::Functional<float>>::operator==),
             py::arg("other"))
        .def(
            "clone",
            (std::unique_ptr<elsa::Functional<float>, std::default_delete<elsa::Functional<float>>>(
                elsa::Cloneable<elsa::Functional<float>>::*)()
                 const)(&elsa::Cloneable<elsa::Functional<float>>::clone));

    py::class_<elsa::Functional<float>, elsa::Cloneable<elsa::Functional<float>>> Functionalf(
        m, "Functionalf");
    Functionalf
        .def("getGradient",
             (elsa::DataContainer<float>(elsa::Functional<float>::*)(
                 const elsa::DataContainer<float>&))(&elsa::Functional<float>::getGradient),
             py::arg("x"), py::return_value_policy::move)
        .def("getHessian",
             (elsa::LinearOperator<float>(elsa::Functional<float>::*)(
                 const elsa::DataContainer<float>&))(&elsa::Functional<float>::getHessian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Functional<float>::*) ()
                  const)(&elsa::Functional<float>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getResidual",
             (const elsa::Residual<float>& (elsa::Functional<float>::*) ()
                  const)(&elsa::Functional<float>::getResidual),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (float(elsa::Functional<float>::*)(const elsa::DataContainer<float>&))(
                 &elsa::Functional<float>::evaluate),
             py::arg("x"))
        .def("getGradient",
             (void(elsa::Functional<float>::*)(const elsa::DataContainer<float>&,
                                               elsa::DataContainer<float>&))(
                 &elsa::Functional<float>::getGradient),
             py::arg("x"), py::arg("result"));

    m.attr("Functional") = m.attr("Functionalf");

    py::class_<elsa::Cloneable<elsa::Functional<double>>> CloneableFunctionald(
        m, "CloneableFunctionald");
    CloneableFunctionald
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Functional<double>>::*)(const elsa::Functional<double>&)
                  const)(&elsa::Cloneable<elsa::Functional<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Functional<double>>::*)(const elsa::Functional<double>&)
                  const)(&elsa::Cloneable<elsa::Functional<double>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Functional<double>,
                                       std::default_delete<elsa::Functional<double>>>(
                          elsa::Cloneable<elsa::Functional<double>>::*)()
                           const)(&elsa::Cloneable<elsa::Functional<double>>::clone));

    py::class_<elsa::Functional<double>, elsa::Cloneable<elsa::Functional<double>>> Functionald(
        m, "Functionald");
    Functionald
        .def("getGradient",
             (elsa::DataContainer<double>(elsa::Functional<double>::*)(
                 const elsa::DataContainer<double>&))(&elsa::Functional<double>::getGradient),
             py::arg("x"), py::return_value_policy::move)
        .def("getHessian",
             (elsa::LinearOperator<double>(elsa::Functional<double>::*)(
                 const elsa::DataContainer<double>&))(&elsa::Functional<double>::getHessian),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Functional<double>::*) ()
                  const)(&elsa::Functional<double>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getResidual",
             (const elsa::Residual<double>& (elsa::Functional<double>::*) ()
                  const)(&elsa::Functional<double>::getResidual),
             py::return_value_policy::reference_internal)
        .def("evaluate",
             (double(elsa::Functional<double>::*)(const elsa::DataContainer<double>&))(
                 &elsa::Functional<double>::evaluate),
             py::arg("x"))
        .def("getGradient",
             (void(elsa::Functional<double>::*)(const elsa::DataContainer<double>&,
                                                elsa::DataContainer<double>&))(
                 &elsa::Functional<double>::getGradient),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::Cloneable<elsa::Functional<thrust::complex<float>>>> CloneableFunctionalcf(
        m, "CloneableFunctionalcf");
    CloneableFunctionalcf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::*)(
                 const elsa::Functional<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::*)(
                 const elsa::Functional<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Functional<thrust::complex<float>>,
                              std::default_delete<elsa::Functional<thrust::complex<float>>>>(
                 elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::*)()
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<float>>>::clone));

    py::class_<elsa::Functional<thrust::complex<float>>,
               elsa::Cloneable<elsa::Functional<thrust::complex<float>>>>
        Functionalcf(m, "Functionalcf");
    Functionalcf
        .def("getGradient",
             (elsa::DataContainer<thrust::complex<float>>(
                 elsa::Functional<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&))(
                 &elsa::Functional<thrust::complex<float>>::getGradient),
             py::arg("x"), py::return_value_policy::move)
        .def("getHessian",
             (elsa::LinearOperator<thrust::complex<float>>(
                 elsa::Functional<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&))(
                 &elsa::Functional<thrust::complex<float>>::getHessian),
             py::arg("x"), py::return_value_policy::move)
        .def("evaluate",
             (thrust::complex<float>(elsa::Functional<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&))(
                 &elsa::Functional<thrust::complex<float>>::evaluate),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Functional<thrust::complex<float>>::*) ()
                  const)(&elsa::Functional<thrust::complex<float>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getResidual",
             (const elsa::Residual<thrust::complex<float>>& (
                 elsa::Functional<thrust::complex<float>>::*) ()
                  const)(&elsa::Functional<thrust::complex<float>>::getResidual),
             py::return_value_policy::reference_internal)
        .def("getGradient",
             (void(elsa::Functional<thrust::complex<float>>::*)(
                 const elsa::DataContainer<thrust::complex<float>>&,
                 elsa::DataContainer<thrust::complex<float>>&))(
                 &elsa::Functional<thrust::complex<float>>::getGradient),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::Cloneable<elsa::Functional<thrust::complex<double>>>> CloneableFunctionalcd(
        m, "CloneableFunctionalcd");
    CloneableFunctionalcd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::*)(
                 const elsa::Functional<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::*)(
                 const elsa::Functional<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Functional<thrust::complex<double>>,
                              std::default_delete<elsa::Functional<thrust::complex<double>>>>(
                 elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::*)()
                  const)(&elsa::Cloneable<elsa::Functional<thrust::complex<double>>>::clone));

    py::class_<elsa::Functional<thrust::complex<double>>,
               elsa::Cloneable<elsa::Functional<thrust::complex<double>>>>
        Functionalcd(m, "Functionalcd");
    Functionalcd
        .def("getGradient",
             (elsa::DataContainer<thrust::complex<double>>(
                 elsa::Functional<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&))(
                 &elsa::Functional<thrust::complex<double>>::getGradient),
             py::arg("x"), py::return_value_policy::move)
        .def("getHessian",
             (elsa::LinearOperator<thrust::complex<double>>(
                 elsa::Functional<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&))(
                 &elsa::Functional<thrust::complex<double>>::getHessian),
             py::arg("x"), py::return_value_policy::move)
        .def("evaluate",
             (thrust::complex<double>(elsa::Functional<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&))(
                 &elsa::Functional<thrust::complex<double>>::evaluate),
             py::arg("x"), py::return_value_policy::move)
        .def("getDomainDescriptor",
             (const elsa::DataDescriptor& (elsa::Functional<thrust::complex<double>>::*) ()
                  const)(&elsa::Functional<thrust::complex<double>>::getDomainDescriptor),
             py::return_value_policy::reference_internal)
        .def("getResidual",
             (const elsa::Residual<thrust::complex<double>>& (
                 elsa::Functional<thrust::complex<double>>::*) ()
                  const)(&elsa::Functional<thrust::complex<double>>::getResidual),
             py::return_value_policy::reference_internal)
        .def("getGradient",
             (void(elsa::Functional<thrust::complex<double>>::*)(
                 const elsa::DataContainer<thrust::complex<double>>&,
                 elsa::DataContainer<thrust::complex<double>>&))(
                 &elsa::Functional<thrust::complex<double>>::getGradient),
             py::arg("x"), py::arg("result"));

    py::class_<elsa::L0PseudoNorm<float>, elsa::Functional<float>> L0PseudoNormf(m,
                                                                                 "L0PseudoNormf");
    L0PseudoNormf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<float>&>(), py::arg("residual"));

    m.attr("L0PseudoNorm") = m.attr("L0PseudoNormf");

    py::class_<elsa::L0PseudoNorm<double>, elsa::Functional<double>> L0PseudoNormd(m,
                                                                                   "L0PseudoNormd");
    L0PseudoNormd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<double>&>(), py::arg("residual"));

    py::class_<elsa::L1Norm<float>, elsa::Functional<float>> L1Normf(m, "L1Normf");
    L1Normf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<float>&>(), py::arg("residual"));

    m.attr("L1Norm") = m.attr("L1Normf");

    py::class_<elsa::L1Norm<double>, elsa::Functional<double>> L1Normd(m, "L1Normd");
    L1Normd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<double>&>(), py::arg("residual"));

    py::class_<elsa::L1Norm<thrust::complex<float>>, elsa::Functional<thrust::complex<float>>>
        L1Normcf(m, "L1Normcf");
    L1Normcf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<thrust::complex<float>>&>(), py::arg("residual"));

    py::class_<elsa::L1Norm<thrust::complex<double>>, elsa::Functional<thrust::complex<double>>>
        L1Normcd(m, "L1Normcd");
    L1Normcd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<thrust::complex<double>>&>(), py::arg("residual"));

    py::class_<elsa::L2NormPow2<float>, elsa::Functional<float>> L2NormPow2f(m, "L2NormPow2f");
    L2NormPow2f.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Residual<float>&>(), py::arg("residual"));

    m.attr("L2NormPow2") = m.attr("L2NormPow2f");

    py::class_<elsa::L2NormPow2<double>, elsa::Functional<double>> L2NormPow2d(m, "L2NormPow2d");
    L2NormPow2d.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Residual<double>&>(), py::arg("residual"));

    py::class_<elsa::L2NormPow2<thrust::complex<float>>, elsa::Functional<thrust::complex<float>>>
        L2NormPow2cf(m, "L2NormPow2cf");
    L2NormPow2cf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&,
                      const elsa::DataContainer<thrust::complex<float>>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Residual<thrust::complex<float>>&>(), py::arg("residual"));

    py::class_<elsa::L2NormPow2<thrust::complex<double>>, elsa::Functional<thrust::complex<double>>>
        L2NormPow2cd(m, "L2NormPow2cd");
    L2NormPow2cd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&,
                      const elsa::DataContainer<thrust::complex<double>>&>(),
             py::arg("A"), py::arg("b"))
        .def(py::init<const elsa::Residual<thrust::complex<double>>&>(), py::arg("residual"));

    py::class_<elsa::WeightedL2NormPow2<float>, elsa::Functional<float>> WeightedL2NormPow2f(
        m, "WeightedL2NormPow2f");
    WeightedL2NormPow2f
        .def("getWeightingOperator",
             (const elsa::Scaling<float>& (elsa::WeightedL2NormPow2<float>::*) ()
                  const)(&elsa::WeightedL2NormPow2<float>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Residual<float>&, const elsa::Scaling<float>&>(),
             py::arg("residual"), py::arg("weightingOp"))
        .def(py::init<const elsa::Scaling<float>&>(), py::arg("weightingOp"));

    m.attr("WeightedL2NormPow2") = m.attr("WeightedL2NormPow2f");

    py::class_<elsa::WeightedL2NormPow2<double>, elsa::Functional<double>> WeightedL2NormPow2d(
        m, "WeightedL2NormPow2d");
    WeightedL2NormPow2d
        .def("getWeightingOperator",
             (const elsa::Scaling<double>& (elsa::WeightedL2NormPow2<double>::*) ()
                  const)(&elsa::WeightedL2NormPow2<double>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Residual<double>&, const elsa::Scaling<double>&>(),
             py::arg("residual"), py::arg("weightingOp"))
        .def(py::init<const elsa::Scaling<double>&>(), py::arg("weightingOp"));

    py::class_<elsa::WeightedL2NormPow2<thrust::complex<float>>,
               elsa::Functional<thrust::complex<float>>>
        WeightedL2NormPow2cf(m, "WeightedL2NormPow2cf");
    WeightedL2NormPow2cf
        .def("getWeightingOperator",
             (const elsa::Scaling<thrust::complex<float>>& (
                 elsa::WeightedL2NormPow2<thrust::complex<float>>::*) ()
                  const)(&elsa::WeightedL2NormPow2<thrust::complex<float>>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Residual<thrust::complex<float>>&,
                      const elsa::Scaling<thrust::complex<float>>&>(),
             py::arg("residual"), py::arg("weightingOp"))
        .def(py::init<const elsa::Scaling<thrust::complex<float>>&>(), py::arg("weightingOp"));

    py::class_<elsa::WeightedL2NormPow2<thrust::complex<double>>,
               elsa::Functional<thrust::complex<double>>>
        WeightedL2NormPow2cd(m, "WeightedL2NormPow2cd");
    WeightedL2NormPow2cd
        .def("getWeightingOperator",
             (const elsa::Scaling<thrust::complex<double>>& (
                 elsa::WeightedL2NormPow2<thrust::complex<double>>::*) ()
                  const)(&elsa::WeightedL2NormPow2<thrust::complex<double>>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::Residual<thrust::complex<double>>&,
                      const elsa::Scaling<thrust::complex<double>>&>(),
             py::arg("residual"), py::arg("weightingOp"))
        .def(py::init<const elsa::Scaling<thrust::complex<double>>&>(), py::arg("weightingOp"));

    py::class_<elsa::LInfNorm<float>, elsa::Functional<float>> LInfNormf(m, "LInfNormf");
    LInfNormf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<float>&>(), py::arg("residual"));

    m.attr("LInfNorm") = m.attr("LInfNormf");

    py::class_<elsa::LInfNorm<double>, elsa::Functional<double>> LInfNormd(m, "LInfNormd");
    LInfNormd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<double>&>(), py::arg("residual"));

    py::class_<elsa::LInfNorm<thrust::complex<float>>, elsa::Functional<thrust::complex<float>>>
        LInfNormcf(m, "LInfNormcf");
    LInfNormcf.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<thrust::complex<float>>&>(), py::arg("residual"));

    py::class_<elsa::LInfNorm<thrust::complex<double>>, elsa::Functional<thrust::complex<double>>>
        LInfNormcd(m, "LInfNormcd");
    LInfNormcd.def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::Residual<thrust::complex<double>>&>(), py::arg("residual"));

    py::class_<elsa::Huber<float>, elsa::Functional<float>> Huberf(m, "Huberf");
    Huberf
        .def(py::init<const elsa::DataDescriptor&, float>(), py::arg("domainDescriptor"),
             py::arg("delta") = static_cast<float>(1.000000e-06))
        .def(py::init<const elsa::Residual<float>&, float>(), py::arg("residual"),
             py::arg("delta") = static_cast<float>(1.000000e-06));

    m.attr("Huber") = m.attr("Huberf");

    py::class_<elsa::Huber<double>, elsa::Functional<double>> Huberd(m, "Huberd");
    Huberd
        .def(py::init<const elsa::DataDescriptor&, float>(), py::arg("domainDescriptor"),
             py::arg("delta") = static_cast<float>(1.000000e-06))
        .def(py::init<const elsa::Residual<double>&, float>(), py::arg("residual"),
             py::arg("delta") = static_cast<float>(1.000000e-06));

    py::class_<elsa::PseudoHuber<float>, elsa::Functional<float>> PseudoHuberf(m, "PseudoHuberf");
    PseudoHuberf
        .def(py::init<const elsa::DataDescriptor&, float>(), py::arg("domainDescriptor"),
             py::arg("delta") = static_cast<float>(1.000000e+00))
        .def(py::init<const elsa::Residual<float>&, float>(), py::arg("residual"),
             py::arg("delta") = static_cast<float>(1.000000e+00));

    m.attr("PseudoHuber") = m.attr("PseudoHuberf");

    py::class_<elsa::PseudoHuber<double>, elsa::Functional<double>> PseudoHuberd(m, "PseudoHuberd");
    PseudoHuberd
        .def(py::init<const elsa::DataDescriptor&, float>(), py::arg("domainDescriptor"),
             py::arg("delta") = static_cast<float>(1.000000e+00))
        .def(py::init<const elsa::Residual<double>&, float>(), py::arg("residual"),
             py::arg("delta") = static_cast<float>(1.000000e+00));

    py::class_<elsa::Quadric<float>, elsa::Functional<float>> Quadricf(m, "Quadricf");
    Quadricf
        .def("getGradientExpression",
             (const elsa::LinearResidual<float>& (elsa::Quadric<float>::*) ()
                  const)(&elsa::Quadric<float>::getGradientExpression),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<float>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<float>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::DataContainer<float>&>(),
             py::arg("A"), py::arg("b"));

    m.attr("Quadric") = m.attr("Quadricf");

    py::class_<elsa::Quadric<double>, elsa::Functional<double>> Quadricd(m, "Quadricd");
    Quadricd
        .def("getGradientExpression",
             (const elsa::LinearResidual<double>& (elsa::Quadric<double>::*) ()
                  const)(&elsa::Quadric<double>::getGradientExpression),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<double>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<double>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::DataContainer<double>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::Quadric<thrust::complex<float>>, elsa::Functional<thrust::complex<float>>>
        Quadriccf(m, "Quadriccf");
    Quadriccf
        .def("getGradientExpression",
             (const elsa::LinearResidual<thrust::complex<float>>& (
                 elsa::Quadric<thrust::complex<float>>::*) ()
                  const)(&elsa::Quadric<thrust::complex<float>>::getGradientExpression),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<thrust::complex<float>>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&,
                      const elsa::DataContainer<thrust::complex<float>>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::Quadric<thrust::complex<double>>, elsa::Functional<thrust::complex<double>>>
        Quadriccd(m, "Quadriccd");
    Quadriccd
        .def("getGradientExpression",
             (const elsa::LinearResidual<thrust::complex<double>>& (
                 elsa::Quadric<thrust::complex<double>>::*) ()
                  const)(&elsa::Quadric<thrust::complex<double>>::getGradientExpression),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<thrust::complex<double>>&>(), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&>(), py::arg("domainDescriptor"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&>(), py::arg("A"))
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&,
                      const elsa::DataContainer<thrust::complex<double>>&>(),
             py::arg("A"), py::arg("b"));

    py::class_<elsa::EmissionLogLikelihood<float>, elsa::Functional<float>> EmissionLogLikelihoodf(
        m, "EmissionLogLikelihoodf");
    EmissionLogLikelihoodf
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<float>&>(),
             py::arg("domainDescriptor"), py::arg("y"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("r"))
        .def(py::init<const elsa::Residual<float>&, const elsa::DataContainer<float>&>(),
             py::arg("residual"), py::arg("y"))
        .def(py::init<const elsa::Residual<float>&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("residual"), py::arg("y"), py::arg("r"));

    m.attr("EmissionLogLikelihood") = m.attr("EmissionLogLikelihoodf");

    py::class_<elsa::EmissionLogLikelihood<double>, elsa::Functional<double>>
        EmissionLogLikelihoodd(m, "EmissionLogLikelihoodd");
    EmissionLogLikelihoodd
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<double>&>(),
             py::arg("domainDescriptor"), py::arg("y"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("r"))
        .def(py::init<const elsa::Residual<double>&, const elsa::DataContainer<double>&>(),
             py::arg("residual"), py::arg("y"))
        .def(py::init<const elsa::Residual<double>&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("residual"), py::arg("y"), py::arg("r"));

    py::class_<elsa::TransmissionLogLikelihood<float>, elsa::Functional<float>>
        TransmissionLogLikelihoodf(m, "TransmissionLogLikelihoodf");
    TransmissionLogLikelihoodf
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&, const elsa::DataContainer<float>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("b"), py::arg("r"))
        .def(py::init<const elsa::Residual<float>&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("residual"), py::arg("y"), py::arg("b"))
        .def(py::init<const elsa::Residual<float>&, const elsa::DataContainer<float>&,
                      const elsa::DataContainer<float>&, const elsa::DataContainer<float>&>(),
             py::arg("residual"), py::arg("y"), py::arg("b"), py::arg("r"));

    m.attr("TransmissionLogLikelihood") = m.attr("TransmissionLogLikelihoodf");

    py::class_<elsa::TransmissionLogLikelihood<double>, elsa::Functional<double>>
        TransmissionLogLikelihoodd(m, "TransmissionLogLikelihoodd");
    TransmissionLogLikelihoodd
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("b"))
        .def(py::init<const elsa::DataDescriptor&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&, const elsa::DataContainer<double>&>(),
             py::arg("domainDescriptor"), py::arg("y"), py::arg("b"), py::arg("r"))
        .def(py::init<const elsa::Residual<double>&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("residual"), py::arg("y"), py::arg("b"))
        .def(py::init<const elsa::Residual<double>&, const elsa::DataContainer<double>&,
                      const elsa::DataContainer<double>&, const elsa::DataContainer<double>&>(),
             py::arg("residual"), py::arg("y"), py::arg("b"), py::arg("r"));

    py::class_<elsa::Cloneable<elsa::Constraint<float>>> CloneableConstraintf(
        m, "CloneableConstraintf");
    CloneableConstraintf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Constraint<float>>::*)(const elsa::Constraint<float>&)
                  const)(&elsa::Cloneable<elsa::Constraint<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Constraint<float>>::*)(const elsa::Constraint<float>&)
                  const)(&elsa::Cloneable<elsa::Constraint<float>>::operator==),
             py::arg("other"))
        .def(
            "clone",
            (std::unique_ptr<elsa::Constraint<float>, std::default_delete<elsa::Constraint<float>>>(
                elsa::Cloneable<elsa::Constraint<float>>::*)()
                 const)(&elsa::Cloneable<elsa::Constraint<float>>::clone));

    py::class_<elsa::Constraint<float>, elsa::Cloneable<elsa::Constraint<float>>> Constraintf(
        m, "Constraintf");
    Constraintf
        .def("getDataVectorC",
             (const elsa::DataContainer<float>& (elsa::Constraint<float>::*) ()
                  const)(&elsa::Constraint<float>::getDataVectorC),
             py::return_value_policy::reference_internal)
        .def("getOperatorA",
             (const elsa::LinearOperator<float>& (elsa::Constraint<float>::*) ()
                  const)(&elsa::Constraint<float>::getOperatorA),
             py::return_value_policy::reference_internal)
        .def("getOperatorB",
             (const elsa::LinearOperator<float>& (elsa::Constraint<float>::*) ()
                  const)(&elsa::Constraint<float>::getOperatorB),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::LinearOperator<float>&, const elsa::LinearOperator<float>&,
                      const elsa::DataContainer<float>&>(),
             py::arg("A"), py::arg("B"), py::arg("c"));

    m.attr("Constraint") = m.attr("Constraintf");

    py::class_<elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>> CloneableConstraintcf(
        m, "CloneableConstraintcf");
    CloneableConstraintcf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::*)(
                 const elsa::Constraint<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::*)(
                 const elsa::Constraint<thrust::complex<float>>&)
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Constraint<thrust::complex<float>>,
                              std::default_delete<elsa::Constraint<thrust::complex<float>>>>(
                 elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::*)()
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>::clone));

    py::class_<elsa::Constraint<thrust::complex<float>>,
               elsa::Cloneable<elsa::Constraint<thrust::complex<float>>>>
        Constraintcf(m, "Constraintcf");
    Constraintcf
        .def("getDataVectorC",
             (const elsa::DataContainer<thrust::complex<float>>& (
                 elsa::Constraint<thrust::complex<float>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<float>>::getDataVectorC),
             py::return_value_policy::reference_internal)
        .def("getOperatorA",
             (const elsa::LinearOperator<thrust::complex<float>>& (
                 elsa::Constraint<thrust::complex<float>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<float>>::getOperatorA),
             py::return_value_policy::reference_internal)
        .def("getOperatorB",
             (const elsa::LinearOperator<thrust::complex<float>>& (
                 elsa::Constraint<thrust::complex<float>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<float>>::getOperatorB),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::LinearOperator<thrust::complex<float>>&,
                      const elsa::LinearOperator<thrust::complex<float>>&,
                      const elsa::DataContainer<thrust::complex<float>>&>(),
             py::arg("A"), py::arg("B"), py::arg("c"));

    py::class_<elsa::Cloneable<elsa::Constraint<double>>> CloneableConstraintd(
        m, "CloneableConstraintd");
    CloneableConstraintd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Constraint<double>>::*)(const elsa::Constraint<double>&)
                  const)(&elsa::Cloneable<elsa::Constraint<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Constraint<double>>::*)(const elsa::Constraint<double>&)
                  const)(&elsa::Cloneable<elsa::Constraint<double>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::Constraint<double>,
                                       std::default_delete<elsa::Constraint<double>>>(
                          elsa::Cloneable<elsa::Constraint<double>>::*)()
                           const)(&elsa::Cloneable<elsa::Constraint<double>>::clone));

    py::class_<elsa::Constraint<double>, elsa::Cloneable<elsa::Constraint<double>>> Constraintd(
        m, "Constraintd");
    Constraintd
        .def("getDataVectorC",
             (const elsa::DataContainer<double>& (elsa::Constraint<double>::*) ()
                  const)(&elsa::Constraint<double>::getDataVectorC),
             py::return_value_policy::reference_internal)
        .def("getOperatorA",
             (const elsa::LinearOperator<double>& (elsa::Constraint<double>::*) ()
                  const)(&elsa::Constraint<double>::getOperatorA),
             py::return_value_policy::reference_internal)
        .def("getOperatorB",
             (const elsa::LinearOperator<double>& (elsa::Constraint<double>::*) ()
                  const)(&elsa::Constraint<double>::getOperatorB),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::LinearOperator<double>&, const elsa::LinearOperator<double>&,
                      const elsa::DataContainer<double>&>(),
             py::arg("A"), py::arg("B"), py::arg("c"));

    py::class_<elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>> CloneableConstraintcd(
        m, "CloneableConstraintcd");
    CloneableConstraintcd
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::*)(
                 const elsa::Constraint<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::*)(
                 const elsa::Constraint<thrust::complex<double>>&)
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::operator==),
             py::arg("other"))
        .def("clone",
             (std::unique_ptr<elsa::Constraint<thrust::complex<double>>,
                              std::default_delete<elsa::Constraint<thrust::complex<double>>>>(
                 elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::*)()
                  const)(&elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>::clone));

    py::class_<elsa::Constraint<thrust::complex<double>>,
               elsa::Cloneable<elsa::Constraint<thrust::complex<double>>>>
        Constraintcd(m, "Constraintcd");
    Constraintcd
        .def("getDataVectorC",
             (const elsa::DataContainer<thrust::complex<double>>& (
                 elsa::Constraint<thrust::complex<double>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<double>>::getDataVectorC),
             py::return_value_policy::reference_internal)
        .def("getOperatorA",
             (const elsa::LinearOperator<thrust::complex<double>>& (
                 elsa::Constraint<thrust::complex<double>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<double>>::getOperatorA),
             py::return_value_policy::reference_internal)
        .def("getOperatorB",
             (const elsa::LinearOperator<thrust::complex<double>>& (
                 elsa::Constraint<thrust::complex<double>>::*) ()
                  const)(&elsa::Constraint<thrust::complex<double>>::getOperatorB),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::LinearOperator<thrust::complex<double>>&,
                      const elsa::LinearOperator<thrust::complex<double>>&,
                      const elsa::DataContainer<thrust::complex<double>>&>(),
             py::arg("A"), py::arg("B"), py::arg("c"));

    py::class_<elsa::WeightedL1Norm<float>, elsa::Functional<float>> WeightedL1Normf(
        m, "WeightedL1Normf");
    WeightedL1Normf
        .def("getWeightingOperator",
             (const elsa::DataContainer<float>& (elsa::WeightedL1Norm<float>::*) ()
                  const)(&elsa::WeightedL1Norm<float>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<float>&>(), py::arg("weightingOp"))
        .def(py::init<const elsa::Residual<float>&, const elsa::DataContainer<float>&>(),
             py::arg("residual"), py::arg("weightingOp"));

    m.attr("WeightedL1Norm") = m.attr("WeightedL1Normf");

    py::class_<elsa::WeightedL1Norm<double>, elsa::Functional<double>> WeightedL1Normd(
        m, "WeightedL1Normd");
    WeightedL1Normd
        .def("getWeightingOperator",
             (const elsa::DataContainer<double>& (elsa::WeightedL1Norm<double>::*) ()
                  const)(&elsa::WeightedL1Norm<double>::getWeightingOperator),
             py::return_value_policy::reference_internal)
        .def(py::init<const elsa::DataContainer<double>&>(), py::arg("weightingOp"))
        .def(py::init<const elsa::Residual<double>&, const elsa::DataContainer<double>&>(),
             py::arg("residual"), py::arg("weightingOp"));

    elsa::FunctionalsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_functionals, m)
{
    add_definitions_pyelsa_functionals(m);
}
