#include "Backtrace.h"

#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sstream>

namespace elsa::detail
{

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

    std::string addrToString(const void* addr)
    {
        std::ostringstream out;
        out << "[" << addr << "]";
        return out.str();
    }

    std::string symbolName(const void* addr, bool require_exact_addr, bool no_pure_addrs)
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
            auto elements = static_cast<size_t>(::backtrace(buffer.data(), buff_size));
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

    void Backtrace::getSymbols(const std::function<void(const backtrace_symbol*)>& cb,
                               bool reversed) const
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

} // namespace elsa
