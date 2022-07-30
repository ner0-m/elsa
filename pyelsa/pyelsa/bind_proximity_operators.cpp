#include <pybind11/pybind11.h>

#include "HardThresholding.h"
#include "ProximityOperator.h"
#include "SoftThresholding.h"

#include "hints/proximity_operators_hints.cpp"

namespace py = pybind11;

void add_definitions_pyelsa_proximity_operators(py::module& m)
{
    py::class_<elsa::Cloneable<elsa::ProximityOperator<float>>> CloneableProximityOperatorf(
        m, "CloneableProximityOperatorf");
    CloneableProximityOperatorf
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::ProximityOperator<float>>::*)(
                 const elsa::ProximityOperator<float>&)
                  const)(&elsa::Cloneable<elsa::ProximityOperator<float>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::ProximityOperator<float>>::*)(
                 const elsa::ProximityOperator<float>&)
                  const)(&elsa::Cloneable<elsa::ProximityOperator<float>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::ProximityOperator<float>,
                                       std::default_delete<elsa::ProximityOperator<float>>>(
                          elsa::Cloneable<elsa::ProximityOperator<float>>::*)()
                           const)(&elsa::Cloneable<elsa::ProximityOperator<float>>::clone));

    py::class_<elsa::ProximityOperator<float>, elsa::Cloneable<elsa::ProximityOperator<float>>>
        ProximityOperatorf(m, "ProximityOperatorf");
    ProximityOperatorf
        .def("apply",
             (elsa::DataContainer<float>(elsa::ProximityOperator<float>::*)(
                 const elsa::DataContainer<float>&, elsa::geometry::Threshold<float>)
                  const)(&elsa::ProximityOperator<float>::apply),
             py::arg("v"), py::arg("t"), py::return_value_policy::move)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::ProximityOperator<float>::*) ()
                  const)(&elsa::ProximityOperator<float>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::ProximityOperator<float>::*)(
                 const elsa::DataContainer<float>&, elsa::geometry::Threshold<float>,
                 elsa::DataContainer<float>&) const)(&elsa::ProximityOperator<float>::apply),
             py::arg("v"), py::arg("t"), py::arg("prox"));

    m.attr("ProximityOperator") = m.attr("ProximityOperatorf");

    py::class_<elsa::Cloneable<elsa::ProximityOperator<double>>> CloneableProximityOperatord(
        m, "CloneableProximityOperatord");
    CloneableProximityOperatord
        .def("__ne__",
             (bool(elsa::Cloneable<elsa::ProximityOperator<double>>::*)(
                 const elsa::ProximityOperator<double>&)
                  const)(&elsa::Cloneable<elsa::ProximityOperator<double>>::operator!=),
             py::arg("other"))
        .def("__eq__",
             (bool(elsa::Cloneable<elsa::ProximityOperator<double>>::*)(
                 const elsa::ProximityOperator<double>&)
                  const)(&elsa::Cloneable<elsa::ProximityOperator<double>>::operator==),
             py::arg("other"))
        .def("clone", (std::unique_ptr<elsa::ProximityOperator<double>,
                                       std::default_delete<elsa::ProximityOperator<double>>>(
                          elsa::Cloneable<elsa::ProximityOperator<double>>::*)()
                           const)(&elsa::Cloneable<elsa::ProximityOperator<double>>::clone));

    py::class_<elsa::ProximityOperator<double>, elsa::Cloneable<elsa::ProximityOperator<double>>>
        ProximityOperatord(m, "ProximityOperatord");
    ProximityOperatord
        .def("apply",
             (elsa::DataContainer<double>(elsa::ProximityOperator<double>::*)(
                 const elsa::DataContainer<double>&, elsa::geometry::Threshold<double>)
                  const)(&elsa::ProximityOperator<double>::apply),
             py::arg("v"), py::arg("t"), py::return_value_policy::move)
        .def("getRangeDescriptor",
             (const elsa::DataDescriptor& (elsa::ProximityOperator<double>::*) ()
                  const)(&elsa::ProximityOperator<double>::getRangeDescriptor),
             py::return_value_policy::reference_internal)
        .def("apply",
             (void(elsa::ProximityOperator<double>::*)(
                 const elsa::DataContainer<double>&, elsa::geometry::Threshold<double>,
                 elsa::DataContainer<double>&) const)(&elsa::ProximityOperator<double>::apply),
             py::arg("v"), py::arg("t"), py::arg("prox"));

    py::class_<elsa::SoftThresholding<float>, elsa::ProximityOperator<float>> SoftThresholdingf(
        m, "SoftThresholdingf");
    SoftThresholdingf.def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    m.attr("SoftThresholding") = m.attr("SoftThresholdingf");

    py::class_<elsa::SoftThresholding<double>, elsa::ProximityOperator<double>> SoftThresholdingd(
        m, "SoftThresholdingd");
    SoftThresholdingd.def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    py::class_<elsa::HardThresholding<float>, elsa::ProximityOperator<float>> HardThresholdingf(
        m, "HardThresholdingf");
    HardThresholdingf.def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    m.attr("HardThresholding") = m.attr("HardThresholdingf");

    py::class_<elsa::HardThresholding<double>, elsa::ProximityOperator<double>> HardThresholdingd(
        m, "HardThresholdingd");
    HardThresholdingd.def(py::init<const elsa::DataDescriptor&>(), py::arg("descriptor"));

    elsa::ProximityOperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_proximity_operators, m)
{
    add_definitions_pyelsa_proximity_operators(m);
}
