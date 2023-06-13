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
    void assert_msg(const char* expr_str, bool expr, const char* file, int line, const char* msg);
} // namespace elsa::detail

/*
  implementation of ENSURE - which is like an assert but "extensible"
  ENSURE(condition)
  ENSURE(condition, errormessage)

  currently calls assert_msg above.
  alternatively we could throw a backtrace capturing elsa-exception here.
*/
#ifndef ENSURE

#define ELSA_EXPRTEXT(expr) #expr

#define ENSURE(...)                                                                 \
    do {                                                                            \
        ::elsa::detail::assert_msg(ELSA_EXPRTEXT(ELSA_ENS_FIRST(__VA_ARGS__)),      \
                                   ELSA_ENS_FIRST(__VA_ARGS__), __FILE__, __LINE__, \
                                   ELSA_ENS_REST(__VA_ARGS__));                     \
    } while (0)

/*
 * expands to the first argument
 * Modified for MSVC using the technique by Jeff Walden
 * https://stackoverflow.com/a/9338429
 */
#define ELSA_PP_GLUE(macro, args) macro args
#define ELSA_ENS_FIRST(...) ELSA_PP_GLUE(ELSA_ENS_FIRST_HELPER, (__VA_ARGS__, throwaway))
#define ELSA_ENS_FIRST_HELPER(first, ...) (first)

/*
 * Standard alternative to GCC's ##__VA_ARGS__ trick (Richard Hansen)
 * http://stackoverflow.com/a/11172679/4742108
 *
 * If there's only one argument, expands to 'nullptr'
 * If there is more than one argument, expands to everything but
 * the first argument. Only supports up to 2 arguments but can be trivially expanded.
 *
 * We could extend this to support arbitrary assert message formatting, with streams, ...
 */
#define ELSA_ENS_REST(...) \
    ELSA_PP_GLUE(ELSA_ENS_REST_HELPER(ELSA_ENS_NUM(__VA_ARGS__)), (__VA_ARGS__))
#define ELSA_ENS_REST_HELPER(qty) ELSA_ENS_REST_HELPER1(qty)
#define ELSA_ENS_REST_HELPER1(qty) ELSA_ENS_REST_HELPER2(qty)
#define ELSA_ENS_REST_HELPER2(qty) ELSA_ENS_REST_HELPER_##qty
#define ELSA_ENS_REST_HELPER_ONE(first) nullptr
#define ELSA_ENS_REST_HELPER_TWOORMORE(first, ...) __VA_ARGS__
#define ELSA_ENS_NUM(...) ELSA_ENS_NUM_IMPL((__VA_ARGS__, TWOORMORE, ONE, throwaway))
#define ELSA_ENS_NUM_IMPL(args) ELSA_ENS_SELECT_2ND args
#define ELSA_ENS_SELECT_2ND(a1, a2, a3, ...) a3
#endif // ENSURE

#define ELSA_VERIFY ENSURE
#define ELSA_TODO() ELSA_VERIFY(false, "TODO")
