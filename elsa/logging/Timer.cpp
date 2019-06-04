#include "Timer.h"

namespace elsa
{
    template <class Duration, class Clock>
    Timer<Duration, Clock>::Timer(std::string name, std::string message)
        : _start{Clock::now()}, _name{std::move(name)}, _message{std::move(message)}
    {}

    template <class Duration, class Clock>
    Timer<Duration, Clock>::~Timer()
    {
        typename Clock::time_point stop = Clock::now();
        auto timeElapsed = std::to_string(std::chrono::duration_cast<Duration>(stop - _start).count());
        Logger::get("Timing")->info("Execution of {}: {} took {} ms", _name, _message, timeElapsed);
    }

    // ------------------------------------------
    // explicit template instantiation
    template class Timer<std::chrono::milliseconds, std::chrono::system_clock>;

} // namespace elsa
