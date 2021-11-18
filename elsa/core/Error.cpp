#include "Error.h"

#include <sstream>

#include "backward.hpp"

namespace elsa
{
    namespace detail
    {
        static std::string get_trace(std::string&& msg)
        {
            namespace bw = backward;

            std::ostringstream stream;

            stream << msg << "\n\n";

            bw::StackTrace stackTrace;
            bw::TraceResolver resolver;
            stackTrace.load_here();
            resolver.load_stacktrace(stackTrace);

            bw::Printer printer;
            printer.color_mode = bw::ColorMode::always; // I always want pretty colors :^)
            printer.print(stackTrace, stream);

            return stream.str();
        }

        BaseError::BaseError(std::string msg) : std::runtime_error(get_trace(std::move(msg))) {}

        std::ostream& operator<<(std::ostream& os, const BaseError& err)
        {
            os << err.what();
            return os;
        }
    } // namespace detail

    Error::Error(const std::string& msg) : BaseError(msg) {}

    InternalError::InternalError(const std::string& msg) : BaseError(msg) {}

    LogicError::LogicError(const std::string& msg) : BaseError(msg) {}

    InvalidArgumentError::InvalidArgumentError(const std::string& msg) : BaseError(msg) {}

    BadCastError::BadCastError(const std::string& msg) : BaseError(msg) {}

} // namespace elsa
