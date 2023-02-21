#pragma once

#include <pybind11/pybind11.h>
#include "DataContainer.h"
#include "Complex.h"
#include "ProximalOperator.h"

#include <iostream>

PYBIND11_MAKE_OPAQUE(elsa::DataContainer<float>);
PYBIND11_MAKE_OPAQUE(elsa::DataContainer<double>);
PYBIND11_MAKE_OPAQUE(elsa::DataContainer<elsa::complex<float>>);
PYBIND11_MAKE_OPAQUE(elsa::DataContainer<elsa::complex<double>>);
PYBIND11_MAKE_OPAQUE(elsa::DataContainer<elsa::index_t>);

/// C++ struct which can bind to a ProximalOperator.
/// This is used for Python types, which have the same interface as ProximalOperator.
/// The PyBind11 custom caster tries to convert a type to a proximal operator. We rely on duck
/// typing, i.e. if a type has a method with the correct name, that'll work. The
/// functions are extracted and stored in this class, and this is passed on to the C++ side
template <class data_t>
struct PyProximalOperator {
    pybind11::function fn;

    PyProximalOperator(pybind11::function fn) : fn(fn) {}

    elsa::DataContainer<data_t> apply(const elsa::DataContainer<data_t>& x, data_t t) const
    {
        return pybind11::cast<elsa::DataContainer<data_t>>(std::move(fn(x, t)));
    }

    void apply(const elsa::DataContainer<data_t>& x, data_t t,
               elsa::DataContainer<data_t>& out) const
    {
        fn(x, data_t{t}, out);
    }
};

namespace PYBIND11_NAMESPACE
{
    namespace detail
    {
        template <class data_t>
        struct type_caster<elsa::ProximalOperator<data_t>> {
            using value_conv = make_caster<data_t>;

        public:
            PYBIND11_TYPE_CASTER(elsa::ProximalOperator<data_t>, const_name("ProximalOperator[")
                                                                     + value_conv::name
                                                                     + const_name("]"));

            bool load(handle src, bool)
            {
                if (!hasattr(src, "apply")) {
                    std::cout << "source has no attribute print\n";
                    std::cout << std::string(str(src)) << "\n";
                    return false;
                }

                auto attr = getattr(src, "apply");

                if (!attr) {
                    std::cout << "Attribute invalid\n";
                    return false;
                }

                auto fn = function(attr);
                value = elsa::ProximalOperator<data_t>(PyProximalOperator<data_t>(fn));

                /* Ensure return code was OK (to avoid out-of-range errors etc) */
                return !PyErr_Occurred();
            }

            static handle cast(elsa::ProximalOperator<data_t> src, return_value_policy /* policy */,
                               handle /* parent */)
            {

                object obj;

                cpp_function fn1([=](const elsa::DataContainer<data_t>& x) { return src.apply(x); },
                                 "apply", is_method(obj), sibling(getattr(obj, "apply", none())));

                cpp_function fn2([=](const elsa::DataContainer<data_t>& x,
                                     elsa::DataContainer<data_t>& out) { src.apply(x, out); },
                                 "apply", is_method(obj), sibling(getattr(obj, "apply", none())));

                obj.attr(fn1.name()) = fn1;
                obj.attr(fn2.name()) = fn2;

                return obj.release();
            }
        };
    } // namespace detail
} // namespace PYBIND11_NAMESPACE
