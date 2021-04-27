#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class WaveletsHints : ModuleHints
    {

        struct LoggerWavelets : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerWavelets>(m, "logger_pyelsa_wavelets")
                .def_static("setLevel", &LoggerWavelets::setLevel)
                .def_static("enableFileLogging", &LoggerWavelets::enableFileLogging)
                .def_static("flush", &LoggerWavelets::flush);
        }
    };
#endif
} // namespace elsa
