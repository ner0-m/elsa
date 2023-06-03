#pragma once

#include "memory_resource/ContiguousVector.h"
#include "memory_resource/AllocationHint.h"
#include "memory_resource/CacheResource.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/RegionResource.h"
#include "memory_resource/LoggingResource.h"
#include "memory_resource/SyncResource.h"
#include "memory_resource/UniversalResource.h"
#include "memory_resource/HostStandardResource.h"

#include "DisableWarnings.h"

DISABLE_WARNING_PUSH
DISABLE_WARNING_SIGN_CONVERSION
#include <thrust/universal_vector.h>
DISABLE_WARNING_POP

namespace elsa
{
    /* inherit from the class to ensure ContiguousStorage<T> can be used to access the explicit
     *  constructors, instead of writing the entire ContigousVector instantiation */
    template <class T>
    class ContiguousStorage final
        : public mr::ContiguousVector<T, mr::type_tags::uninitialized, thrust::universal_ptr,
                                      thrust::universal_ptr>
    {
    public:
        using mr::ContiguousVector<T, mr::type_tags::uninitialized, thrust::universal_ptr,
                                   thrust::universal_ptr>::ContiguousVector;
    };
} // namespace elsa