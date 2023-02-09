#include "Error.h"

#include <sstream>

#include "Backtrace.h"

namespace elsa
{
    constexpr const char* runtime_error_message = "polymorphic elsa error, catch by reference!";

    Error::Error(std::string msg, bool generate_backtrace, bool store_cause)
        : std::runtime_error{runtime_error_message}, backtrace{nullptr}, msg{std::move(msg)}
    {

        if (generate_backtrace) {
            this->backtrace = std::make_shared<Backtrace>();
            this->backtrace->analyze();
        }

        if (store_cause) {
            this->storeCause();
        }
    }

    std::string Error::str() const
    {
        return this->msg;
    }

    const char* Error::what() const noexcept
    {
        this->what_cache = this->str();
        return this->what_cache.c_str();
    }

    void Error::storeCause()
    {
        if (not std::current_exception()) {
            return;
        }

        try {
            throw;
        } catch (Error& cause) {
            cause.trimBacktrace();
            this->cause = std::current_exception();
        } catch (...) {
            this->cause = std::current_exception();
        }
    }

    void Error::trimBacktrace()
    {
        if (this->backtrace) {
            this->backtrace->trimToCurrentStackFrame();
        }
    }

    void Error::rethrowCause() const
    {
        if (this->cause) {
            std::rethrow_exception(this->cause);
        }
    }

    std::string Error::typeName() const
    {
        return detail::symbolDemangle(typeid(*this).name());
    }

    Backtrace* Error::getBacktrace() const
    {
        return this->backtrace.get();
    }

    const std::string& Error::getMsg() const
    {
        return this->msg;
    }

    std::ostream& operator<<(std::ostream& os, const Error& e)
    {
        // output the exception cause
        bool had_a_cause = true;
        try {
            e.rethrowCause();
            had_a_cause = false;
        } catch (Error& cause) {
            os << cause << std::endl;
        } catch (std::exception& cause) {
            os << detail::symbolDemangle(typeid(cause).name()) << ": " << cause.what() << std::endl;
        }

        if (had_a_cause) {
            os << std::endl
               << "The above exception was the direct cause "
                  "of the following exception:"
               << std::endl
               << std::endl;
        }

        // output the exception backtrace
        auto* bt = e.getBacktrace();
        if (bt != nullptr) {
            os << *bt;
        } else {
            os << "origin:" << std::endl;
        }

        os << e.typeName() << ":" << std::endl;
        os << e.str();

        return os;
    }

    /**
     * Prints a backtrace_symbol object.
     */
    std::ostream& operator<<(std::ostream& os, const backtrace_symbol& bt_sym)
    {
        // imitate the looks of a Python traceback.
        os << " -> ";

        if (bt_sym.functionname.empty()) {
            os << '?';
        } else {
            os << bt_sym.functionname;
        }

        if (bt_sym.pc != nullptr) {
            os << " " << detail::addrToString(bt_sym.pc);
        }

        return os;
    }

    /**
     * Prints an entire Backtrace object.
     */
    std::ostream& operator<<(std::ostream& os, const Backtrace& bt)
    {
        // imitate the looks of a Python traceback.
        os << "Traceback (most recent call last):" << std::endl;

        bt.getSymbols([&os](const backtrace_symbol* symbol) { os << *symbol << std::endl; }, true);

        return os;
    }

    InternalError::InternalError(const std::string& msg) : Error{msg} {}

    LogicError::LogicError(const std::string& msg) : Error{msg} {}

    InvalidArgumentError::InvalidArgumentError(const std::string& msg) : Error{msg} {}

    BadCastError::BadCastError(const std::string& msg) : Error{msg} {}

    NotImplementedError::NotImplementedError(const std::string& msg) : Error{msg} {}

} // namespace elsa
