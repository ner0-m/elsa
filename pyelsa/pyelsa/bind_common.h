#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <string>
#include <complex>
#include <tuple>
#include "LinearOperator.h"
#include "spdlog/fmt/fmt.h"

template <typename type, typename... options>
void add_linear_operator_fns(pybind11::class_<type, options...>& op)
{
    using namespace elsa;
    namespace py = pybind11;
    using T = typename type::value_type;

    const auto& reference_internal = py::return_value_policy::reference_internal;
    const auto& move = py::return_value_policy::move;

    using DCRef = DataContainer<T>&;
    using ConstDCRef = const DataContainer<T>&;

    // clang-format off
    op.def("getDomainDescriptor", &type::getDomainDescriptor, reference_internal)
      .def("getRangeDescriptor", &type::getRangeDescriptor, reference_internal)
      .def("apply", py::overload_cast<ConstDCRef>(&type::apply, py::const_),
                    py::arg("x"), move)
      .def("apply", py::overload_cast<ConstDCRef, DCRef>(&type::apply, py::const_),
                    py::arg("x"), py::arg("Ax"), move)
      .def("applyAdjoint", py::overload_cast<ConstDCRef>(&type::applyAdjoint, py::const_),
                           py::arg("y"), move)
      .def("applyAdjoint", py::overload_cast<ConstDCRef, DCRef>(&type::applyAdjoint, py::const_),
                           py::arg("y"), py::arg("Aty"), move)
      .def("__mul__",
            [](T scalar, const elsa::LinearOperator<T>& op) {
                return scalar * op;
            }, py::is_operator())
      .def("__mul__",
            [](const elsa::LinearOperator<T> &lhs, const elsa::LinearOperator<T>& rhs) {
                return rhs * lhs;
            }, py::is_operator())
      .def("__add__",
            [](const elsa::LinearOperator<T> &lhs, const elsa::LinearOperator<T>& rhs) {
                return lhs + rhs;
            }, py::is_operator())
      ; // <- just here to quickly add and remove lines, without worrying about it
    // clang-format on
}

template <class T, template <class...> class Op>
auto add_derived_single_operator(pybind11::module& m, const std::string& name)
{
    namespace py = pybind11;

    py::class_<Op<T>, elsa::LinearOperator<T>> op(m, name.c_str());
    add_linear_operator_fns(op);
    return op;
}

template <template <class...> class Op>
auto add_derived_operator(pybind11::module& m, const std::string& name)
{
    namespace py = pybind11;

    auto opf = add_derived_single_operator<float, Op>(m, fmt::format("{}f", name).c_str());
    auto opd = add_derived_single_operator<double, Op>(m, fmt::format("{}d", name).c_str());
    auto opcf =
        add_derived_single_operator<std::complex<float>, Op>(m, fmt::format("{}cf", name).c_str());
    auto opcd =
        add_derived_single_operator<std::complex<double>, Op>(m, fmt::format("{}cd", name).c_str());

    // Set the float version as default
    m.attr(name.c_str()) = m.attr(fmt::format("{}f", name).c_str());

    return std::make_tuple(opf, opd, opcf, opcd);
}
