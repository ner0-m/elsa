#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class ProjectorsCUDAHints : ModuleHints
    {

        struct LoggerProjectorsCUDA : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerProjectorsCUDA>(m, "logger_pyelsa_projectors_cuda")
                .def_static("setLevel", &LoggerProjectorsCUDA::setLevel)
                .def_static("enableFileLogging", &LoggerProjectorsCUDA::enableFileLogging)
                .def_static("flush", &LoggerProjectorsCUDA::flush);
        }
    };
#endif
} // namespace elsa
