#include "hints_base.h"

#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
#include "Logger.h"
#endif

#include "WLSSubsetProblem.h"
#include "LinearOperator.h"

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

    template <typename data_t>
    class WLSSubsetProblemHints : public ClassHints<elsa::WLSSubsetProblem<data_t>>
    {
        static std::vector<std::unique_ptr<LinearOperator<data_t>>>
            cloneOpList(std::vector<const LinearOperator<data_t>*>& opList)
        {
            std::vector<std::unique_ptr<LinearOperator<data_t>>> cloneList;
            for (const auto op : opList) {
                cloneList.push_back(op->clone());
            }
            return cloneList;
        }

    public:
        template <typename type_, typename... options>
        static void addCustomMethods(py::class_<type_, options...>& c)
        {
            // all constructors disabled on the Python side as they use unique_ptr
            // create constructor wrappers
            c.def(py::init([](const LinearOperator<data_t>& A, const DataContainer<data_t>& b,
                              std::vector<const LinearOperator<data_t>*>& subsetAs) {
                return std::make_unique<WLSSubsetProblem<data_t>>(A, b, cloneOpList(subsetAs));
            }));
        }
    };

    template class WLSSubsetProblemHints<float>;
    template class WLSSubsetProblemHints<double>;
    template class WLSSubsetProblemHints<complex<float>>;
    template class WLSSubsetProblemHints<complex<double>>;
} // namespace elsa
