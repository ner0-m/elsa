#include "Assertions.h"
#include "elsaDefines.h"

namespace elsa::detail
{
    void assert_msg(const char* expr_str, bool expr, const char* file, int line, const char* msg)
    {
        if (unlikely(not expr)) {
            std::cerr << "Assert failed:";
            if (msg != nullptr) {
                std::cerr << "\t" << msg;
            }
            std::cerr << "\nExpected:\t" << expr_str << "\n"
                      << "Source:\t\t" << file << ", line " << line << "\n";
            abort();
        }
    }
} // namespace elsa::detail
