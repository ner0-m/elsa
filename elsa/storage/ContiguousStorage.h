#pragma once

// Thrust is smart enough to always pick the correct vector for us
#include "thrust/universal_vector.h"

namespace elsa
{
    template <class T>
    using ContiguousStorage = thrust::universal_vector<T>;
}
