#include "Assertions.h"

namespace elsa::detail
{
    void assert_nomsg(const char* expr_str, bool expr, const char* file, int line)
    {
        if (!expr) {
            std::cerr << "Assert failed:\n"
                      << "Expected:\t" << expr_str << "\n"
                      << "Source:\t\t" << file << ", line " << line << "\n";
            abort();
        }
    }

    void assert_msg(const char* expr_str, bool expr, const char* file, int line, const char* msg)
    {
        if (!expr) {
            std::cerr << "Assert failed:\t" << msg << "\n"
                      << "Expected:\t" << expr_str << "\n"
                      << "Source:\t\t" << file << ", line " << line << "\n";
            abort();
        }
    }
} // namespace elsa::detail
