#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class GenratorsHints : ModuleHints
    {

        struct LoggerGenerators : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerGenerators>(m, "logger_pyelsa_generators")
                .def_static("setLevel", &LoggerGenerators::setLevel)
                .def_static("enableFileLogging", &LoggerGenerators::enableFileLogging)
                .def_static("flush", &LoggerGenerators::flush);
        }
    };
#endif
} // namespace elsa
