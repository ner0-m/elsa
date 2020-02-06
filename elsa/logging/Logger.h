#pragma once

#include <map>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>

namespace elsa
{
    /**
     * \brief Logging class for elsa purposes (wrapper around spdlog).
     *
     * \author Maximilian Hornung - initial version
     * \author Tobias Lasser - rewrite
     *
     * This class provides logging for the elsa library. It is a thin wrapper around the spdlog
     * third party library, using the single-threaded versions of the sinks.
     *
     * Logger::get("name") requests a logger (from spdlog) corresponding to "name", which is used
     * just as you would use spdlog directly, i.e. you can use the info(), debug(), warn() etc.
     * routines. For more  details on spdlog, please visit https://github.com/gabime/spdlog.
     *
     * By default, Logger enables console logging. If requested, file logging can enabled in
     * addition via addFileSink. Log levels are set via setLevel (again, they correspond to the
     * spdlog log levels).
     */
    class Logger
    {
    public:
        /// return a logger corresponding to "name" (if not existing, it is created)
        static std::shared_ptr<spdlog::logger> get(std::string name);

        /// available log levels (default: INFO), corresponding to those of spdlog
        enum class LogLevel { TRACE, DEBUG, INFO, WARN, ERR, CRITICAL, OFF };

        /// set the log level for all loggers
        static void setLevel(LogLevel level);

        /// enable file logging
        static void enableFileLogging(std::string filename);

        /// add a given sink pointer to the sink list of a target logger
        static void addCustomSink(spdlog::sink_ptr sink, std::string logger_name);

        /// delete every Logger initialized so far
        static void clearMap();

        /// returns true if the _loggers map is empty
        static bool isEmpty();

    private:
        /// returns the singleton instance of Logger
        static Logger& getInstance();

        /// the log level
        LogLevel _level{LogLevel::INFO};

        /// convert elsa LogLevel to spdlog::level::level_enum
        static spdlog::level::level_enum convertLevelToSpdlog(LogLevel level);

        /// the file name for file logging (if enabled)
        std::unique_ptr<std::string> _fileName{};

        /// map storing the loggers identified by a string
        std::map<std::string, std::shared_ptr<spdlog::logger>> _loggers;

        /// friend the LogGuard class to enable access to private methods
        friend class LogGuard;
    };
} // namespace elsa
