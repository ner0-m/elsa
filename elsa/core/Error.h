#pragma once

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace elsa
{

    /**
     * A single symbol, as determined from a program counter, and returned by
     * Backtrace::getSymbols().
     */
    struct backtrace_symbol {
        std::string functionname; // empty if unknown
        void* pc;                 // nullptr if unknown
    };

    /**
     * Provide execution backtrace information through getSymbols().
     */
    class Backtrace
    {
    public:
        Backtrace() = default;

        virtual ~Backtrace() = default;

        /**
         * Analyzes the current stack, and stores the program counter values in
         * this->stack_addrs.
         */
        void analyze();

        /**
         * Provide the names for all stack frames via the callback.
         *
         * The most recent call is returned last (alike Python).
         *
         * @param cb
         *    is called for every symbol in the backtrace,
         *    starting with the top-most frame.
         * @param reversed
         *    if true, the most recent call is given last.
         */
        void getSymbols(std::function<void(const backtrace_symbol*)> cb,
                        bool reversed = true) const;

        /**
         * Removes all the lower frames that are also present
         * in the current stack.
         *
         * Designed to be used in catch clauses,
         * to simulate stack trace collection
         * from throw to catch, instead of from throw to the process entry point.
         */
        void trimToCurrentStackFrame();

    protected:
        /**
         * All program counters of this backtrace.
         */
        std::vector<void*> stack_addrs;
    };

    /**
     * Base exception for every error that occurs in elsa.
     *
     * Usage:
     * try {
     *     throw Error{"some info"};
     * }
     * catch (Error &err) {
     *     std::cout << "\x1b[31;1merror:\x1b[m\n"
     *               << err << std::endl;
     * }
     *
     */
    class Error : public std::runtime_error
    {
    public:
        Error(const std::string& msg, bool generate_backtrace = true, bool store_cause = true);

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

} // namespace elsa
