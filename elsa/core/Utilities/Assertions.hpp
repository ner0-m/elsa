#pragma once

#include <iostream>

namespace elsa::detail
{
    void elsa_assert(const char* expr_str, bool expr, const char* file, int line)
    {
        if (!static_cast<bool>(expr)) {
            std::cerr << "Assert failed:\n"
                      << "Expected:\t" << expr_str << "\n"
                      << "Source:\t\t" << file << ", line " << line << "\n";
            // TODO: Will this be caught by backward-cpp? Ensure it
            abort();
        }
    }

    void elsa_assert_msg(const char* expr_str, bool expr, const char* file, int line,
                         const char* msg)
    {
        if (!static_cast<bool>(expr)) {
            std::cerr << "Assert failed:\t" << msg << "\n"
                      << "Expected:\t" << expr_str << "\n"
                      << "Source:\t\t" << file << ", line " << line << "\n";
            // TODO: Will this be caught by backward-cpp? Ensure it
            abort();
        }
    }
} // namespace elsa::detail

// I hate macros but this is just convenience...

#define ELSA_CAT(A, B) A##B
#define ELSA_SELECT(NAME, NUM) ELSA_CAT(NAME##_, NUM)
#define ELSA_GET_COUNT(_1, _2, _3, COUNT, ...) COUNT
#define ELSA_VA_SIZE(...) ELSA_GET_COUNT(__VA_ARGS__, 3, 2, 1)
#define ELSA_VA_SELECT(NAME, ...) ELSA_SELECT(NAME, ELSA_VA_SIZE(__VA_ARGS__))(__VA_ARGS__)

#define ELSA_VERIFY(...) ELSA_VA_SELECT(ELSA_VERIFY, __VA_ARGS__)

#ifdef ELSA_ENABLE_ASSERTIONS
#define ELSA_VERIFY_1(Expr) ::elsa::detail::elsa_assert(#Expr, Expr, __FILE__, __LINE__)
#define ELSA_VERIFY_2(Expr, Msg) \
    ::elsa::detail::elsa_assert_msg(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#define ELSA_VERIFY_1(Expr)
#define ELSA_VERIFY_2(Expr, Msg)
#endif

// This should come in handy as well
#if _MSC_VER && !__INTEL_COMPILER
#define ELSA_UNREACHABLE() __assume(false)
#else
#define ELSA_UNREACHABLE() __builtin_unreachable()
#endif
