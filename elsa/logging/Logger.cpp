#include "Logger.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/sinks/ostream_sink.h>

namespace elsa
{
    void Logger::setLevel(LogLevel level)
    {
        getInstance()._level = level;

        // set level globally (probably superfluous..)
        spdlog::set_level(convertLevelToSpdlog(level));

        // set level for all active loggers
        for (auto& [key, logger] : getInstance()._loggers)
            logger->set_level(convertLevelToSpdlog(level));
    }

    std::shared_ptr<spdlog::sinks::dist_sink_st> Logger::initSinks()
    {
        auto sink = std::make_shared<spdlog::sinks::dist_sink_st>();

        // Add a console output sink
        sink->add_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

        // If a filename is set, also add a file output sink
        if (getInstance()._fileName != "")
            sink->add_sink(
                std::make_shared<spdlog::sinks::basic_file_sink_st>(getInstance()._fileName));

        return sink;
    }

    std::shared_ptr<spdlog::sinks::dist_sink_st> Logger::sinks()
    {
        static auto sink = initSinks();
        return sink;
    }

    std::shared_ptr<spdlog::logger> Logger::get(std::string name)
    {
        // If we don't have sinks setup yet, initialize it

        if (getInstance()._loggers.count(name) == 0) {
            auto newLogger = std::make_shared<spdlog::logger>(name, sinks());
            newLogger->set_level(convertLevelToSpdlog(getInstance()._level));
            getInstance()._loggers[name] = newLogger;
        }

        return getInstance()._loggers[name];
    }

    void Logger::enableFileLogging(std::string filename)
    {
        getInstance()._fileName = std::move(filename);

        sinks()->add_sink(
            std::make_shared<spdlog::sinks::basic_file_sink_st>(getInstance()._fileName));

        // for (auto& [key, logger] : getInstance()._loggers) {}
    }

    void Logger::flush()
    {
        for (auto& [key, logger] : getInstance()._loggers) {
            for (auto sink : logger->sinks())
                sink->flush();
        }
    }

    void Logger::addSink(std::ostream& os)
    {
        sinks()->add_sink(std::make_shared<spdlog::sinks::ostream_sink_st>(os));
        // for (auto& [key, logger] : getInstance()._loggers) {
        //     auto distSink = dynamic_cast<spdlog::sinks::dist_sink_st*>(logger->sinks()[0].get());
        //     if (distSink) {
        //         distSink->add_sink(std::make_shared<spdlog::sinks::ostream_sink_st>(os));
        //     }
        // }
    }

    Logger& Logger::getInstance()
    {
        static Logger instance;
        return instance;
    }

    spdlog::level::level_enum Logger::convertLevelToSpdlog(Logger::LogLevel level)
    {
        switch (level) {
            case LogLevel::TRACE:
                return spdlog::level::trace;
            case LogLevel::DEBUG:
                return spdlog::level::debug;
            case LogLevel::INFO:
                return spdlog::level::info;
            case LogLevel::WARN:
                return spdlog::level::warn;
            case LogLevel::ERR:
                return spdlog::level::err;
            case LogLevel::CRITICAL:
                return spdlog::level::critical;
            case LogLevel::OFF:
                return spdlog::level::off;
            default:
                return spdlog::level::info;
        }
    }

} // namespace elsa
