#pragma once

#include <string>

#include "Logger.h"

namespace elsa
{
    /**
     * @brief simple log guard class, logs a message each on creation and destruction.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - minor modifications
     *
     * This class serves as a guard for logging, simple instantiation will log a starting message,
     * while going out of scope logs an ending message.
     */
    class LogGuard
    {
    public:
        /// constructor taking the name of the logger to log to, the message and the log level of
        /// the message
        LogGuard(std::string loggerName, std::string message,
                 Logger::LogLevel level = Logger::LogLevel::INFO);

        /// destructor outputting the final log message
        ~LogGuard();

    private:
        std::string _loggerName; /// the name of the logger to log to
        std::string _message;    /// the message to log
        Logger::LogLevel _level; /// the log level of the message
    };

} // namespace elsa
