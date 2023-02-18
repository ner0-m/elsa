#include <pybind11/pybind11.h>

#include "DataContainer.h"
#include "ProximalOperator.h"
#include "ProximalL1.h"
#include "ProximalL0.h"

#include "StrongTypes.h"
#include "hints/proximal_operators_hints.cpp"

#include "bind_common.h"

namespace py = pybind11;

namespace detail
{
    /* template <class data_t> */
    /* void add_proximal_op(py::module& m, py::class_<elsa::ProximalL1<data_t>> c) */
    template <class data_t, template <class> class Prox>
    void add_proximal_op(py::module& m, py::class_<Prox<data_t>> c)
    {
        using Self = Prox<data_t>;
        using Threshold = elsa::geometry::Threshold<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;

        c.def(py::init<>());
        c.def("apply",
              py::overload_cast<const DataContainer&, Threshold>(&Self::apply, py::const_));
        c.def("apply", py::overload_cast<const DataContainer&, Threshold, DataContainer&>(
                           &Self::apply, py::const_));
    }
} // namespace detail

void add_definitions_pyelsa_proximal_operators(py::module& m)
{
    py::class_<elsa::ProximalL1<float>> proxL1f(m, "ProximalL1f");
    detail::add_proximal_op(m, proxL1f);

    py::class_<elsa::ProximalL1<double>> proxL1d(m, "ProximalL1d");
    detail::add_proximal_op(m, proxL1d);
    m.attr("ProximalL1") = m.attr("ProximalL1f");

    py::class_<elsa::ProximalL0<float>> proxL0f(m, "ProximalL0f");
    detail::add_proximal_op(m, proxL0f);

    py::class_<elsa::ProximalL0<double>> proxL0d(m, "ProximalL0d");
    detail::add_proximal_op(m, proxL0d);
    m.attr("ProximalL0") = m.attr("ProximalL0f");

    elsa::ProximalOperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_proximal_operators, m)
{
    add_definitions_pyelsa_proximal_operators(m);
}
