#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class SolversHints : ModuleHints
    {

        struct LoggerSolvers : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerSolvers>(m, "logger_pyelsa_solvers")
                .def_static("setLevel", &LoggerSolvers::setLevel)
                .def_static("enableFileLogging", &LoggerSolvers::enableFileLogging)
                .def_static("flush", &LoggerSolvers::flush);
        }
    };
#endif
} // namespace elsa
