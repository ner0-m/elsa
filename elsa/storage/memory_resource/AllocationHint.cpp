#include "AllocationHint.h"

#include "PoolResource.h"
#include "CacheResource.h"
#include "RegionResource.h"

#include <variant>
#include <limits>

namespace elsa::mr::hint
{
    thread_local std::optional<MemoryResource> HINT;

    ScopedMR::ScopedMR(const MemoryResource& hint)
    {
        _previous = std::move(HINT);
        HINT = hint;
    }

    ScopedMR::~ScopedMR()
    {
        HINT = std::move(_previous);
    }

    std::optional<MemoryResource> selectMemoryResource()
    {
        if (!HINT) {
            return std::nullopt;
        }
        return *HINT;
    }
} // namespace elsa::mr::hint