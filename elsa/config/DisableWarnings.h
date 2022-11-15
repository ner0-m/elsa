#pragma once

#if __NVCC__

#define ELSA_MAKE_PRAGMA(X) _Pragma(#X)
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#define ELSA_NV_DIAG_SUPPRESS(X) ELSA_MAKE_PRAGMA(nv_diag_suppress X)
#else
#define ELSA_NV_DIAG_SUPPRESS(X) ELSA_MAKE_PRAGMA(diag_suppress X)
#endif

// Silence warning "__host__ annotation is ignored on a [...] that is explicitly defaulted on its
// first declaration"
ELSA_NV_DIAG_SUPPRESS(20012)
ELSA_NV_DIAG_SUPPRESS(186)

#undef ELSA_NV_DIAG_SUPPRESS
#undef ELSA_MAKE_PRAGMA
#endif

/*
 * Credit go to Jonathan Boccara. This is taken from the blog:
 * https://www.fluentcpp.com/2019/08/30/how-to-disable-a-warning-in-cpp/
 */

#if defined(_MSC_VER)
#define DISABLE_WARNING_PUSH __pragma(warning(push))
#define DISABLE_WARNING_POP __pragma(warning(pop))
#define DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))

#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(4100)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(4505)
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)

#define DO_PRAGMA(X) _Pragma(#X)
#define DISABLE_WARNING_PUSH DO_PRAGMA(GCC diagnostic push)
#define DISABLE_WARNING_POP DO_PRAGMA(GCC diagnostic pop)
#define DISABLE_WARNING(warningName) DO_PRAGMA(GCC diagnostic ignored #warningName)

// clang-format off
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER DISABLE_WARNING(-Wunused-parameter)
#define DISABLE_WARNING_UNREFERENCED_FUNCTION DISABLE_WARNING(-Wunused-function)
#define DISABLE_WARNING_SIGN_CONVERSION DISABLE_WARNING(-Wsign-conversion)
// other warnings you want to deactivate...
// clang-format on
#else
#define DISABLE_WARNING_PUSH
#define DISABLE_WARNING_POP
#define DISABLE_WARNING_UNREFERENCED_FORMAL_PARAMETER
#define DISABLE_WARNING_UNREFERENCED_FUNCTION
// other warnings you want to deactivate...

#endif
