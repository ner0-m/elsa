#include "EDFHandler.h"
#include "MHDHandler.h"

#include <pybind11/pybind11.h>

#include <functional>

namespace elsa
{
    template <typename Class>
    struct ClassHints {
    };

    template <class Return, class... Args>
    using CustomMethod = std::pair<bool, std::function<Return(Args...)>>;

    namespace py = pybind11;

    struct EDFHints : public ClassHints<EDF> {
        constexpr static std::array ignoreMethods = {"read"};

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def_static(
                 "readf", [](std::string filename) { return EDF::read<float>(filename); },
                 py::return_value_policy::move)
                .def_static(
                    "readd", [](std::string filename) { return EDF::read<double>(filename); },
                    py::return_value_policy::move)
                .def_static(
                    "readl", [](std::string filename) { return EDF::read<index_t>(filename); },
                    py::return_value_policy::move);
        }
    };

    struct MHDHints : public ClassHints<MHD> {
        constexpr static std::array ignoreMethods = {"read"};

        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            c.def_static(
                 "readf", [](std::string filename) { return MHD::read<float>(filename); },
                 py::return_value_policy::move)
                .def_static(
                    "readd", [](std::string filename) { return MHD::read<double>(filename); },
                    py::return_value_policy::move)
                .def_static(
                    "readl", [](std::string filename) { return MHD::read<index_t>(filename); },
                    py::return_value_policy::move);
        }
    };
} // namespace elsa