#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProximalOperatorsHints : ModuleHints
    {

        struct LoggerProximalOperators : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProximalOperators>(m, "logger_pyelsa_proximal_operators")
                .def_static("setLevel", &LoggerProximalOperators::setLevel)
                .def_static("enableFileLogging", &LoggerProximalOperators::enableFileLogging)
                .def_static("flush", &LoggerProximalOperators::flush);
        }
    };
#endif
} // namespace elsa
