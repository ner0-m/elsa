#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProjectorsHints : ModuleHints
    {
        struct LoggerProjectors : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProjectors>(m, "logger_pyelsa_projectors")
                .def_static("setLevel", &LoggerProjectors::setLevel)
                .def_static("enableFileLogging", &LoggerProjectors::enableFileLogging)
                .def_static("flush", &LoggerProjectors::flush);
        }
    };
#endif
} // namespace elsa
