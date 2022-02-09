#pragma once

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

#include "spdlog/fmt/fmt.h"

namespace elsa
{

    // fwd-decls for Backtrace.h
    struct backtrace_symbol;
    class Backtrace;

    /**
     * Base exception for every error that occurs in elsa.
     *
     * Usage:
     * try {
     *     throw elsa::Error{"some info"};
     * }
     * catch (elsa::Error &err) {
     *     std::cout << "\x1b[31;1merror:\x1b[m\n"
     *               << err << std::endl;
     * }
     *
     */
    class Error : public std::runtime_error
    {
    public:
        template <typename FormatString, typename... Args>
        explicit Error(const FormatString& fmt, Args&&... args)
            : Error(fmt::format(fmt, std::forward<Args>(args)...))
        {
        }

        Error(std::string msg, bool generate_backtrace = true, bool store_cause = true);

        ~Error() override = default;

        /**
         * String representation of this exception, as
         * specialized by a child exception.
         */
        virtual std::string str() const;

        /**
         * Returns the message's content.
         */
        const char* what() const noexcept override;

        /**
         * Stores a pointer to the currently-handled exception in this->cause.
         */
        void storeCause();

        /**
         * Calls this->backtrace->trimToCurrentStackFrame(),
         * if this->backtrace is not nullptr.
         *
         * Designed to be used in catch clauses, to strip away all those
         * unneeded symbols from program init upwards.
         */
        void trimBacktrace();

        /**
         * Re-throws the exception cause, if the exception has one.
         * Otherwise, does nothing.
         *
         * Use this when handling the exception, to handle the cause.
         */
        void rethrowCause() const;

        /**
         * Get the type name of of the exception.
         * Use it to display the name of a child exception.
         */
        virtual std::string typeName() const;

        /**
         * Return the backtrace where the exception was thrown.
         * nullptr if no backtrace was collected.
         */
        Backtrace* getBacktrace() const;

        /**
         * Directly return the message stored in the exception.
         */
        const std::string& getMsg() const;

    protected:
        /**
         * The (optional) backtrace where the exception came from.
         */
        std::shared_ptr<Backtrace> backtrace;

        /**
         * The error message text.
         */
        std::string msg;

        /**
         * Cached error message text for returning C string via what().
         */
        mutable std::string what_cache;

        /**
         * Re-throw this with rethrowCause().
         */
        std::exception_ptr cause;
    };

    /**
     * Output stream concat for Errors.
     */
    std::ostream& operator<<(std::ostream& os, const Error& e);

    /**
     * Output stream concat for backtrace symbols.
     */
    std::ostream& operator<<(std::ostream& os, const backtrace_symbol& bt_sym);

    /**
     * Output stream concat for backtraces.
     */
    std::ostream& operator<<(std::ostream& os, const Backtrace& bt);

    /**
     * Internal Error, thrown when some interal sanity check failed.
     */
    class InternalError : public Error
    {
    public:
        InternalError(const std::string& msg);
    };

    /**
     * Faulty logic due to violation of expected preconditions.
     */
    class LogicError : public Error
    {
    public:
        LogicError(const std::string& msg);
    };

    /**
     * User gave something wrong to our function.
     */
    class InvalidArgumentError : public Error
    {
    public:
        InvalidArgumentError(const std::string& msg);
    };

    /**
     * Could not cast to the given type
     */
    class BadCastError : public Error
    {
    public:
        BadCastError(const std::string& msg);
    };

} // namespace elsa
