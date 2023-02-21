#include <pybind11/pybind11.h>

#include "DataContainer.h"
#include "ProximalOperator.h"
#include "ProximalL1.h"
#include "ProximalL0.h"
#include "ProximalBoxConstraint.h"

#include "StrongTypes.h"
#include "hints/proximal_operators_hints.cpp"

#include "bind_common.h"

namespace py = pybind11;

template <class data_t>
elsa::DataContainer<data_t> proxapply(elsa::ProximalOperator<data_t> prox,
                                      const elsa::DataContainer<data_t>& x, data_t t)
{
    return prox.apply(x, t);
}

template <class data_t>
void proxapplyout(elsa::ProximalOperator<data_t> prox, const elsa::DataContainer<data_t>& x,
                  data_t t, elsa::DataContainer<data_t>& out)
{
    prox.apply(x, t, out);
}

namespace detail
{
    template <class data_t, template <class> class Prox>
    void add_proximal_op(py::module& m, py::class_<Prox<data_t>> c)
    {
        using Self = Prox<data_t>;
        using DataContainer = elsa::DataContainer<data_t>;

        c.def(py::init<>());
        c.def("apply", py::overload_cast<const DataContainer&, data_t>(&Self::apply, py::const_),
              py::return_value_policy::move);
        c.def("apply", py::overload_cast<const DataContainer&, data_t, DataContainer&>(&Self::apply,
                                                                                       py::const_));

        // py::implicitly_convertible<Prox<data_t>, elsa::ProximalOperator<data_t>>();
    }
} // namespace detail

void add_definitions_pyelsa_proximal_operators(py::module& m)
{
    py::class_<elsa::ProximalOperator<float>> proxf(m, "ProximalOperatorf");
    detail::add_proximal_op(m, proxf);

    py::class_<elsa::ProximalOperator<double>> proxd(m, "ProximalOperatord");
    detail::add_proximal_op(m, proxd);
    m.attr("ProximalOperator") = m.attr("ProximalOperatorf");

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

    py::class_<elsa::ProximalBoxConstraint<float>> proxboxf(m, "ProximalBoxConstraintf");
    detail::add_proximal_op(m, proxboxf);
    proxboxf.def(py::init<>());
    proxboxf.def(py::init<float>(), py::arg("lower"));
    proxboxf.def(py::init<float, float>(), py::arg("lower"), py::arg("upper"));

    py::class_<elsa::ProximalBoxConstraint<double>> proxboxd(m, "ProximalBoxConstraintd");
    detail::add_proximal_op(m, proxboxd);
    proxboxd.def(py::init<>());
    proxboxd.def(py::init<double>(), py::arg("lower"));
    proxboxd.def(py::init<double, double>(), py::arg("lower"), py::arg("upper"));
    m.attr("ProximalBoxConstraint") = m.attr("ProximalBoxConstraintf");

    m.def("proxapply", &proxapply<float>, py::return_value_policy::move);
    m.def("proxapply", &proxapply<double>, py::return_value_policy::move);

    m.def("proxapplyout", &proxapplyout<float>);
    m.def("proxapplyout", &proxapplyout<double>);

    elsa::ProximalOperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_proximal_operators, m)
{
    add_definitions_pyelsa_proximal_operators(m);
}
