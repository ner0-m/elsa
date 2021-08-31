#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

namespace elsa
{
    namespace py = pybind11;

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
    class TasksHints : ModuleHints
    {

        struct LoggerTasks : Logger {
        };

    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::class_<LoggerTasks>(m, "logger_pyelsa_tasks")
                .def_static("setLevel", &LoggerTasks::setLevel)
                .def_static("enableFileLogging", &LoggerTasks::enableFileLogging)
                .def_static("flush", &LoggerTasks::flush);
        }
    };
#endif
} // namespace elsa
