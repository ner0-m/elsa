#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProximityOperatorsHints : ModuleHints
    {

        struct LoggerProximityOperators : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProximityOperators>(m, "logger_pyelsa_proximity_operators")
                .def_static("setLevel", &LoggerProximityOperators::setLevel)
                .def_static("enableFileLogging", &LoggerProximityOperators::enableFileLogging)
                .def_static("flush", &LoggerProximityOperators::flush);
        }
    };
#endif
} // namespace elsa
