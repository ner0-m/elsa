#pragma once

#if defined __NVCC__
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
