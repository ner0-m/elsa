#pragma once

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    template <class T>
    using ContiguousStorage = thrust::universal_vector<T>;
}
