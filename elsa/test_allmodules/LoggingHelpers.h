#pragma once

#include "elsaDefines.h"
#include "Logger.h"

#include "spdlog/formatter.h"

#include <string_view>

namespace elsa
{
    struct ConsoleLogging {
        static void logHeader()
        {
            // Create logger
            auto logger = Logger::get("Benchmark");

            // Log it
            Logger::setLevel(Logger::LogLevel::INFO);
            logger->set_pattern("%v");
            logger->info("| {:^17} | {:^9} | {:^21} | {:^41} | {:^15} |", "General", "", "Solver",
                         "Time (in s)", "Error");
            logger->info(
                "| {:^3} | {:^4} | {:^5}| {:^9} | {:^13} | {:^5} | {:^8} | {:^8} | {:^8} | "
                "{:^8} | {:^6} | {:^6} | ",
                "Dim", "Size", "Reps", "Projector", "Name", "Iters", "Mean", "StdDev", "95% low",
                "95% high", "Abs", "Rel");
            Logger::setLevel(Logger::LogLevel::OFF);
        }

        static void logLaps(int dim, int size, std::size_t benchIters, std::string_view opName,
                            std::string_view solName, std::size_t noIters, real_t timeMean,
                            real_t timeStddev, real_t timeLower, real_t timeUpper,
                            real_t absErrMean, real_t relErrMean)
        {
            // Log output of this iterations
            Logger::setLevel(Logger::LogLevel::INFO);
            auto log = Logger::get(std::string{solName});
            log->set_pattern("%v");
            log->info("| {:>3} | {:>4} | {:>4} | {:>9} | {:>13} | {:>5} | {:>8.4f} | {:>8.4f} | "
                      "{:>8.4f} | {:>8.4f} | {:>6.4f} | "
                      "{:>6.4f} |",
                      dim, size, benchIters, opName, solName, noIters, timeMean, timeStddev,
                      timeLower, timeUpper, absErrMean, relErrMean);
            Logger::setLevel(Logger::LogLevel::OFF);
        }
    };

    class CarriageReturnFormatter : public spdlog::formatter
    {
    public:
        auto format(const spdlog::details::log_msg& msg, spdlog::memory_buf_t& dest)
            -> void override
        {
            // Append normal message
            dest.append(msg.payload.data(), msg.payload.data() + msg.payload.size());

            // Now append '\r' instead of '\n'
            std::string eol = "\r";
            dest.append(eol.data(), eol.data() + eol.size());
        }

        auto clone() const -> std::unique_ptr<spdlog::formatter> override
        {
            return spdlog::details::make_unique<CarriageReturnFormatter>();
        }
    };
} // namespace elsa
