#pragma once

#include <string>
#include <functional>
#include <vector>

namespace elsa::detail
{

    /**
     * Demangles a symbol name.
     *
     * On failure, the mangled symbol name is returned.
     */
    std::string symbolDemangle(const char* symbol);

    /**
     * Convert a pointer address to string.
     */
    std::string addrToString(const void* addr);

    /**
     * Return the demangled symbol name for a given code address.
     */
    std::string symbolName(const void* addr, bool require_exact_addr = true,
                           bool no_pure_addrs = false);

} // namespace elsa::detail

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
        void getSymbols(const std::function<void(const backtrace_symbol*)>& cb,
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
} // namespace elsa
