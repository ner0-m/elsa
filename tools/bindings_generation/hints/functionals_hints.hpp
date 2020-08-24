#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class FunctionalsHints : ModuleHints
    {

        struct LoggerFunctionals : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerFunctionals>(m, "logger_pyelsa_functionals")
                .def_static("setLevel", &LoggerFunctionals::setLevel)
                .def_static("enableFileLogging", &LoggerFunctionals::enableFileLogging)
                .def_static("flush", &LoggerFunctionals::flush);
        }
    };
#endif
} // namespace elsa
