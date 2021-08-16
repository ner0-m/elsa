#include "Timer.h"

namespace elsa
{
    Timer::Timer(std::string caller, std::string method)
        : _caller{std::move(caller)}, _method{std::move(method)}
    {
    }

    Timer::~Timer() { Logger::get(_caller)->debug("{} took {:.3}s", _method, _watch); }
} // namespace elsa
