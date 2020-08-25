#include "hints_base.h"

#include "BlockLinearOperator.h"
#include "Logger.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace elsa
{
    namespace py = pybind11;

    class OperatorsHints : ModuleHints
    {
#ifndef ELSA_BINDINGS_IN_SINGLE_MODULE
        struct LoggerOperators : Logger {
        };
#endif
    public:
        static void addCustomFunctions(py::module& m)
        {
            // expose Logger class
            py::enum_<Logger::LogLevel>(m, "LogLevel")
                .value("OFF", Logger::LogLevel::OFF)
                .value("TRACE", Logger::LogLevel::TRACE)
                .value("DEBUG", Logger::LogLevel::DEBUG)
                .value("INFO", Logger::LogLevel::INFO)
                .value("WARN", Logger::LogLevel::WARN)
                .value("ERR", Logger::LogLevel::ERR)
                .value("CRITICAL", Logger::LogLevel::CRITICAL);

#ifdef ELSA_BINDINGS_IN_SINGLE_MODULE
            py::class_<Logger>(m, "Logger")
                .def_static("setLevel", &Logger::setLevel)
                .def_static("enableFileLogging", &Logger::enableFileLogging)
                .def_static("flush", &Logger::flush);
#else
            // uses a somewhat ugly typename on the Python side to simplify generation of the
            // top-level Logger
            py::class_<LoggerOperators>(m, "logger_pyelsa_operators")
                .def_static("setLevel", &LoggerOperators::setLevel)
                .def_static("enableFileLogging", &LoggerOperators::enableFileLogging)
                .def_static("flush", &LoggerOperators::flush);
#endif
        }
    };

    template <typename data_t>
    class BlockLinearOperatorHints : public ClassHints<elsa::BlockLinearOperator<data_t>>
    {
        using BlockType = typename BlockLinearOperator<data_t>::BlockType;
        using OperatorList = typename BlockLinearOperator<data_t>::OperatorList;

        static OperatorList cloneOpList(std::vector<const LinearOperator<data_t>*>& opList)
        {
            OperatorList cloneList;
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
            c.def(py::init(
                      [](std::vector<const LinearOperator<data_t>*>& opList, BlockType blockType) {
                          return std::make_unique<BlockLinearOperator<data_t>>(cloneOpList(opList),
                                                                               blockType);
                      }))
                .def(py::init([](const DataDescriptor& domainDescriptor,
                                 const DataDescriptor& rangeDescriptor,
                                 std::vector<const LinearOperator<data_t>*>& opList,
                                 BlockType blockType) {
                    return std::make_unique<BlockLinearOperator<data_t>>(
                        domainDescriptor, rangeDescriptor, cloneOpList(opList), blockType);
                }));
        }
    };

    template class BlockLinearOperatorHints<float>;
    template class BlockLinearOperatorHints<double>;
    template class BlockLinearOperatorHints<std::complex<float>>;
    template class BlockLinearOperatorHints<std::complex<double>>;
} // namespace elsa
