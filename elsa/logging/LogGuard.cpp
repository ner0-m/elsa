#include "LogGuard.h"

namespace elsa
{
    LogGuard::LogGuard(std::string loggerName, std::string message, Logger::LogLevel level)
        : _loggerName{std::move(loggerName)}, _message{std::move(message)}, _level{level}
    {
        auto logger = Logger::get(_loggerName);
        logger->log(Logger::convertLevelToSpdlog(_level), "Start: " + _message);
    }

    LogGuard::~LogGuard()
    {
        auto logger = Logger::get(_loggerName);
        logger->log(Logger::convertLevelToSpdlog(_level), "End: " + _message);
    }

} // namespace elsa
