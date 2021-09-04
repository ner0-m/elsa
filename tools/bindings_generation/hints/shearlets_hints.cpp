#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ShearletsHints : ModuleHints
    {

        struct LoggerShearlets : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerShearlets>(m, "logger_pyelsa_shearlets")
                .def_static("setLevel", &LoggerShearlets::setLevel)
                .def_static("enableFileLogging", &LoggerShearlets::enableFileLogging)
                .def_static("flush", &LoggerShearlets::flush);
        }
    };
#endif
} // namespace elsa
