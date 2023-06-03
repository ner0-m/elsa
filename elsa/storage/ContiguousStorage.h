#pragma once

#include "memory_resource/ContiguousVector.h"
#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    template <class T>
    using ContiguousStorage = mr::ContiguousVector<T, mr::type_tags::uninitialized,
                                                   thrust::universal_ptr, thrust::universal_ptr>;
} // namespace elsa