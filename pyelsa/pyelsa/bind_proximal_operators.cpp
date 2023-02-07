#include <pybind11/pybind11.h>

#include "DataContainer.h"
#include "ProximalOperator.h"
#include "ProximalIdentity.h"
#include "ProximalL1.h"
#include "ProximalL0.h"
#include "ProximalL2Squared.h"
#include "CombinedProximal.h"

#include "StrongTypes.h"
#include "hints/proximal_operators_hints.cpp"

namespace py = pybind11;

namespace detail
{
    template <class data_t, template <class> class Prox>
    void add_proximal_op(py::class_<Prox<data_t>> c)
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

void add_prox_identity(py::module& m)
{
    py::class_<elsa::ProximalIdentity<float>> proxf(m, "ProximalIdentityf");
    detail::add_proximal_op(proxf);

    py::class_<elsa::ProximalIdentity<double>> proxd(m, "ProximalIdentityd");
    detail::add_proximal_op(proxd);

    py::implicitly_convertible<elsa::ProximalIdentity<float>, elsa::ProximalOperator<float>>();
    py::implicitly_convertible<elsa::ProximalIdentity<double>, elsa::ProximalOperator<double>>();

    m.attr("ProximalIdentity") = m.attr("ProximalIdentityf");
}

void add_prox_l0(py::module& m)
{
    py::class_<elsa::ProximalL0<float>> proxL0f(m, "ProximalL0f");
    detail::add_proximal_op(proxL0f);

    py::class_<elsa::ProximalL0<double>> proxL0d(m, "ProximalL0d");
    detail::add_proximal_op(proxL0d);

    py::implicitly_convertible<elsa::ProximalL0<float>, elsa::ProximalOperator<float>>();
    py::implicitly_convertible<elsa::ProximalL0<double>, elsa::ProximalOperator<double>>();

    m.attr("ProximalL0") = m.attr("ProximalL0f");
}

void add_prox_l1(py::module& m)
{
    py::class_<elsa::ProximalL1<float>> proxf(m, "ProximalL1f");
    detail::add_proximal_op(proxf);

    py::class_<elsa::ProximalL1<double>> proxd(m, "ProximalL1d");
    detail::add_proximal_op(proxd);

    py::implicitly_convertible<elsa::ProximalL1<float>, elsa::ProximalOperator<float>>();
    py::implicitly_convertible<elsa::ProximalL1<double>, elsa::ProximalOperator<double>>();

    m.attr("ProximalL1") = m.attr("ProximalL1f");
}

void add_prox_l2squared(py::module& m)
{
    py::class_<elsa::ProximalL2Squared<float>> proxf(m, "ProximalL2Squaredf");
    detail::add_proximal_op(proxf);
    proxf.def(py::init<const elsa::DataContainer<float>&>(), py::arg("b"));

    py::class_<elsa::ProximalL2Squared<double>> proxd(m, "ProximalL2Squaredd");
    proxd.def(py::init<const elsa::DataContainer<double>&>(), py::arg("b"));
    detail::add_proximal_op(proxd);

    py::implicitly_convertible<elsa::ProximalL2Squared<float>, elsa::ProximalOperator<float>>();
    py::implicitly_convertible<elsa::ProximalL2Squared<double>, elsa::ProximalOperator<double>>();

    m.attr("ProximalL2Squared") = m.attr("ProximalL2Squaredf");
}

namespace detail
{
    template <class data_t>
    void add_prox_combined(py::module& m, const char* name)
    {
        using Prox = elsa::ProximalOperator<data_t>;

        py::class_<elsa::CombinedProximal<data_t>> proxf(m, name);
        proxf.def(py::init<Prox>());
        proxf.def(py::init<Prox, Prox>());
        proxf.def(py::init<Prox, Prox, Prox>());
        proxf.def(py::init<Prox, Prox, Prox, Prox>());
        proxf.def(py::init<Prox, Prox, Prox, Prox, Prox>());
        detail::add_proximal_op(proxf);

        py::implicitly_convertible<elsa::CombinedProximal<data_t>, Prox>();
    }
} // namespace detail
void add_prox_combined(py::module& m)
{
    detail::add_prox_combined<float>(m, "CombinedProximalf");
    detail::add_prox_combined<double>(m, "CombinedProximald");

    m.attr("CombinedProximal") = m.attr("CombinedProximalf");
}

void add_prox_op(py::module& m)
{
    py::class_<elsa::ProximalOperator<float>> proxf(m, "ProximalOperatorf");

    proxf.def(py::init<elsa::ProximalL0<float>>());
    proxf.def(py::init<elsa::ProximalL1<float>>());
    proxf.def(py::init<elsa::ProximalL2Squared<float>>());
    proxf.def(py::init<elsa::ProximalIdentity<float>>());
    proxf.def(py::init<elsa::CombinedProximal<float>>());

    detail::add_proximal_op(proxf);

    py::class_<elsa::ProximalOperator<double>> proxd(m, "ProximalOperatord");

    proxd.def(py::init<elsa::ProximalL0<double>>());
    proxd.def(py::init<elsa::ProximalL1<double>>());
    proxd.def(py::init<elsa::ProximalL2Squared<double>>());
    proxd.def(py::init<elsa::ProximalIdentity<double>>());
    proxd.def(py::init<elsa::CombinedProximal<double>>());

    detail::add_proximal_op(proxd);

    m.attr("ProximalOperator") = m.attr("ProximalOperatorf");
}

void add_definitions_pyelsa_proximal_operators(py::module& m)
{
    add_prox_op(m);

    add_prox_identity(m);
    add_prox_l0(m);
    add_prox_l1(m);
    add_prox_l2squared(m);
    add_prox_combined(m);

    elsa::ProximalOperatorsHints::addCustomFunctions(m);
}

PYBIND11_MODULE(pyelsa_proximal_operators, m)
{
    add_definitions_pyelsa_proximal_operators(m);
}
