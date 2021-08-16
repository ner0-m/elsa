#pragma once

#include <chrono>
#include <string>

#include "spdlog/stopwatch.h"

#include "Logger.h"

namespace elsa
{
    /**
     * @brief Timer class to provide easy logging of timing.
     *
     * This class provides logging of timing using the guard pattern. It measures the current time
     * from creation, until destruction and outputs a log message with time elapsed on destruction.
     *
     * @author Matthias Wieczorek - initial code
     * @author Maximilian Hornung - modularization
     * @author Tobias Lasser - minor changes
     * @author David Frank - use spdlog::stopwatch, renaming of variables for expressing intent
     * better
     *
     */
    class Timer
    {
    public:
        /// start the timer, using loggerName as logger, outputting message at end
        Timer(std::string caller, std::string method);

        /// stop the timer and actually output the log message
        ~Timer();

    private:
        spdlog::stopwatch _watch{}; /// stopwatch measuring the time
        std::string _caller;        /// the name of what is running
        std::string _method;        /// the message to output (in addition to timing)
    };

} // namespace elsa
