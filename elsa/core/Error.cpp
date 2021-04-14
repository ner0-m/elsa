#include "Error.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <memory>
#include <sstream>

namespace elsa::detail
{

    /**
     * Demangles a symbol name.
     *
     * On failure, the mangled symbol name is returned.
     */
    std::string symbolDemangle(const char* symbol)
    {
        int status;
        char* buf = abi::__cxa_demangle(symbol, nullptr, nullptr, &status);

        if (status != 0) {
            return symbol;
        } else {
            std::string result{buf};
            ::free(buf);
            return result;
        }
    }

    /**
     * Convert a pointer address to string.
     */
    std::string addrToString(const void* addr)
    {
        std::ostringstream out;
        out << "[" << addr << "]";
        return out.str();
    }

    /**
     * Return the demangled symbol name for a given code address.
     */
    std::string symbolName(const void* addr, bool require_exact_addr = true,
                           bool no_pure_addrs = false)
    {
        Dl_info addr_info;

        if (::dladdr(addr, &addr_info) == 0) {
            // dladdr has... failed.
            return no_pure_addrs ? "" : addrToString(addr);
        } else {
            size_t symbol_offset =
                reinterpret_cast<size_t>(addr) - reinterpret_cast<size_t>(addr_info.dli_saddr);

            if (addr_info.dli_sname == nullptr or (symbol_offset != 0 and require_exact_addr)) {

                return no_pure_addrs ? "" : addrToString(addr);
            }

            if (symbol_offset == 0) {
                // this is our symbol name.
                return symbolDemangle(addr_info.dli_sname);
            } else {
                std::ostringstream out;
                out << symbolDemangle(addr_info.dli_sname) << "+0x" << std::hex << symbol_offset
                    << std::dec;
                return out.str();
            }
        }
    }

} // namespace elsa::detail

namespace elsa
{

    void Backtrace::analyze()
    {
        std::vector<void*> buffer{32};

        // increase buffer size until it's enough
        while (true) {
            int buff_size = static_cast<int>(buffer.size());
            size_t elements = static_cast<size_t>(::backtrace(buffer.data(), buff_size));
            if (elements < buffer.size()) {
                buffer.resize(elements);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }

        for (void* element : buffer) {
            this->stack_addrs.push_back(element);
        }
    }

    void Backtrace::getSymbols(std::function<void(const backtrace_symbol*)> cb, bool reversed) const
    {
        backtrace_symbol symbol;

        if (reversed) {
            for (size_t idx = this->stack_addrs.size(); idx-- > 0;) {
                void* pc = this->stack_addrs[idx];

                symbol.functionname = detail::symbolName(pc, false, true);
                symbol.pc = pc;

                cb(&symbol);
            }
        } else {
            for (void* pc : this->stack_addrs) {
                symbol.functionname = detail::symbolName(pc, false, true);
                symbol.pc = pc;

                cb(&symbol);
            }
        }
    }

    void Backtrace::trimToCurrentStackFrame()
    {
        Backtrace current;
        current.analyze();

        while (not current.stack_addrs.empty() and not this->stack_addrs.empty()) {
            if (this->stack_addrs.back() != current.stack_addrs.back()) {
                break;
            }

            this->stack_addrs.pop_back();
            current.stack_addrs.pop_back();
        }
    }

    constexpr const char* runtime_error_message = "polymorphic elsa error, catch by reference!";

    Error::Error(const std::string& msg, bool generate_backtrace, bool store_cause)
        : std::runtime_error{runtime_error_message}, backtrace{nullptr}, msg{msg}
    {

        if (generate_backtrace) {
            this->backtrace = std::make_shared<Backtrace>();
            this->backtrace->analyze();
        }

        if (store_cause) {
            this->storeCause();
        }
    }

    std::string Error::str() const { return this->msg; }

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

    std::string Error::typeName() const { return detail::symbolDemangle(typeid(*this).name()); }

    Backtrace* Error::getBacktrace() const { return this->backtrace.get(); }

    const std::string& Error::getMsg() const { return this->msg; }

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
        auto bt = e.getBacktrace();
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

} // namespace elsa
