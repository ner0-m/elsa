#pragma once

#include <iostream>

// Based on the standard paper P0627r0 (see
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0627r0.pdf)
#if defined(_MSC_VER)
#define ELSA_UNREACHABLE() __assume(false);
#elif defined(__GNUC__) or defined(__clang__)
// All gcc/clang compilers supporting c++17 have __builtin_unreachable
#define ELSA_UNREACHABLE() __builtin_unreachable()
#else
#include <exception>
#define ELSA_UNREACHABLE() std::terminate()
#endif

namespace elsa::detail
{
    void assert_nomsg(const char* expr_str, bool expr, const char* file, int line);

    void assert_msg(const char* expr_str, bool expr, const char* file, int line, const char* msg);
} // namespace elsa::detail

#define GET_MACRO(_1, _2, NAME, ...) NAME

#define ELSA_VERIFY_IMPL2(Expr, Msg) \
    ::elsa::detail::assert_msg(#Expr, Expr, __FILE__, __LINE__, Msg)
#define ELSA_VERIFY_IMPL1(Expr) ::elsa::detail::assert_nomsg(#Expr, Expr, __FILE__, __LINE__)

#define ELSA_VERIFY(...) GET_MACRO(__VA_ARGS__, ELSA_VERIFY_IMPL2, ELSA_VERIFY_IMPL1)(__VA_ARGS__)

#define ELSA_TODO() ELSA_VERIFY(false, "TODO")
