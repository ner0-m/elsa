#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProblemsHints : ModuleHints
    {
        struct LoggerProblems : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProblems>(m, "logger_pyelsa_problems")
                .def_static("setLevel", &LoggerProblems::setLevel)
                .def_static("enableFileLogging", &LoggerProblems::enableFileLogging)
                .def_static("flush", &LoggerProblems::flush);
        }
    };
#endif
} // namespace elsa
