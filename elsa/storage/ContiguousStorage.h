#pragma once

#include "memory_resource/ContiguousVector.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/UniversalResource.h"

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
// Thrust is smart enough to always pick the correct vector for us
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    namespace detail
    {
        class ConfigMemoryResource
        {
        public:
            ConfigMemoryResource()
            {
                if (!mr::baselineInstanceSet())
                    mr::setBaselineInstance(mr::UniversalResource::make());
            }
        };
        static ConfigMemoryResource _singleton;
    } // namespace detail

    template <class T>
    using ContiguousStorage = mr::ContiguousVector<T, mr::type_tags::complex>;
    // using ContiguousStorage = thrust::universal_vector<T>;
} // namespace elsa