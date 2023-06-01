#pragma once

#include "memory_resource/ContiguousVector.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/ElsaThrustMRAdaptor.h"
#include "memory_resource/UniversalResource.h"
#include "memory_resource/ContiguousWrapper.h"

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    template <class T>
    using ContiguousStorage = mr::ContiguousVector<T, mr::type_tags::uninitialized,
                                                   thrust::universal_ptr, thrust::universal_ptr>;
    // using ContiguousStorage = thrust::universal_vector<T>;
} // namespace elsa