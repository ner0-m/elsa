#pragma once

#include <chrono>
#include <string>

#include "Logger.h"


namespace elsa
{
    /**
     * \brief Timer class to provide easy logging of timing.
     *
     * \author Matthias Wieczorek - initial code
     * \author Maximilian Hornung - modularization
     * \author Tobias Lasser - minor changes
     *
     * This class provides logging of timing using the guard pattern. It stores the current  time
     * upon creation, and outputs a log message with time elapsed on destruction.
     */
     template <class Duration = std::chrono::milliseconds,
               class Clock = std::chrono::system_clock>
     class Timer {
     public:
         /// start the timer, using loggerName as logger, outputting message at end
         Timer(std::string name, std::string message);

         /// stop the timer and actually output the log message
         ~Timer();

     private:
         typename Clock::time_point _start; /// the start time on creation
         std::string _name;                 /// the name of what is running
         std::string _message;              /// the message to output (in addition to timing)
     };

} // namespace elsa
