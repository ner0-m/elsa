#pragma once

#include <pybind11/pybind11.h>

namespace elsa
{
    namespace py = pybind11;
    /**
     * @brief Base class for all class specific hints
     *
     * @tparam Class the class for which hints are provided
     *
     * Class specific hints should be provided in a subclass of ClassHints and shadow the
     * corresponding static methods to provide the desired functionality on the Python side.
     *
     */
    template <typename Class>
    class ClassHints
    {
    public:
        // shadow in sublclass and initialize inline to hide specified methods on the Python side
        constexpr static std::array<std::string, 0> ignoreMethods = {};

        // shadow in subclass to provide support for the Python buffer protocol
        // see https://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html#buffer-protocol
        template <typename type_, typename... options>
        static void exposeBufferInfo(py::class_<type_, options...>& c);

        // shadow in subclass to define any additional methods on the Python side
        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c);
    };

    /**
     * @brief Base class for hints applying to an entire module
     *
     * Define additional global functions and attributes in the subclass
     */
    class ModuleHints
    {
    public:
        // shadow in sublcass to define additional global methods
        static void addCustomFunctions(py::module& m);
    };
} // namespace elsa