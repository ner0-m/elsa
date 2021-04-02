#pragma once

#include <unordered_map>
#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace elsa
{
    /**
     * @brief Logging class for elsa purposes (wrapper around spdlog).
     *
     * @author Maximilian Hornung - initial version
     * @author Tobias Lasser - rewrite
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

        /// flush all loggers
        static void flush();

        /// add a sink that writes to `std::ostream` (useful for testing)
        static void addSink(std::ostream& os);

    private:
        /// returns the singleton instance of Logger
        static Logger& getInstance();

        /// Static function initialising the sinks
        static std::shared_ptr<spdlog::sinks::dist_sink_st> initSinks();

        /// Return the sinks stored for the loggers
        static std::shared_ptr<spdlog::sinks::dist_sink_st> sinks();

        /// the log level
        LogLevel _level{LogLevel::INFO};

        /// convert elsa LogLevel to spdlog::level::level_enum
        static spdlog::level::level_enum convertLevelToSpdlog(LogLevel level);

        /// the file name for file logging (if enabled)
        std::string _fileName{""};

        /// map storing the loggers identified by a string
        std::unordered_map<std::string, std::shared_ptr<spdlog::logger>> _loggers;

        /// friend the LogGuard class to enable access to private methods
        friend class LogGuard;
    };

    namespace detail
    {
        template <typename FormatString, typename... Args>
        void logln(spdlog::level::level_enum lvl, FormatString msg, Args&&... args)
        {
            spdlog::set_pattern("[%^%=7l%$] %v");
            spdlog::log(spdlog::source_loc{}, lvl, msg, std::forward<Args>(args)...);
        }

        template <typename FormatString, typename... Args>
        void logln(spdlog::source_loc loc, spdlog::level::level_enum lvl, const FormatString& msg,
                   Args&&... args)
        {
            spdlog::set_pattern("[%^%=7l%$] [%!@%s:%#] %v");
            spdlog::log(loc, lvl, msg, std::forward<Args>(args)...);
        }
    } // namespace detail

    template <typename FormatString, typename... Args>
    void infoln(const FormatString& msg, Args&&... args)
    {
        ::elsa::detail::logln(spdlog::level::level_enum::info, msg, std::forward<Args>(args)...);
    }

    template <typename FormatString, typename... Args>
    void warnln(const FormatString& msg, Args&&... args)
    {
        ::elsa::detail::logln(spdlog::level::level_enum::warn, msg, std::forward<Args>(args)...);
    }

    template <typename FormatString, typename... Args>
    void dbgln(const FormatString& msg, Args&&... args)
    {
        ::elsa::detail::logln(spdlog::level::level_enum::debug, msg, std::forward<Args>(args)...);
    }

#define INFOLN(...)                                                                \
    ::elsa::detail::logln(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                          spdlog::level::level_enum::info, __VA_ARGS__)

#define WARNLN(...)                                                                \
    ::elsa::detail::logln(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                          spdlog::level::level_enum::warn, __VA_ARGS__)

#define DBGLN(...)                                                                 \
    ::elsa::detail::logln(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                          spdlog::level::level_enum::debug, __VA_ARGS__)
} // namespace elsa
