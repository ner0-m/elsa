#include "AllocationHint.h"

#include "PoolResource.h"
#include "CacheResource.h"
#include "RegionResource.h"

#include "Overloaded.hpp"
#include <variant>
#include <limits>

namespace elsa::mr::hint
{
    thread_local std::unique_ptr<std::variant<MemoryResource, AllocationBehavior>> HINT;

    template <typename H>
    ScopedAllocationHint<H>::ScopedAllocationHint(const H& hint)
    {
        _previous = std::move(HINT);
        HINT = std::make_unique<std::variant<MemoryResource, AllocationBehavior>>(hint);
    }

    template <typename H>
    ScopedAllocationHint<H>::~ScopedAllocationHint()
    {
        HINT = std::move(_previous);
    }

    template class ScopedAllocationHint<MemoryResource>;

    template class ScopedAllocationHint<AllocationBehavior>;

    std::optional<MemoryResource> selectMemoryResource()
    {
        if (!HINT) {
            return std::nullopt;
        }
        return std::visit(
            overloaded{
                [](const MemoryResource& mr) { return mr; },
                [](const AllocationBehavior& behavior) {
                    MemoryResource result;
                    if (behavior.allocThenFullRelease()) {
                        if (behavior.bulk()) {
                            auto config = CacheResourceConfig::defaultConfig();
                            if (behavior.getSizeHint() != 0) {
                                config.setMaxCacheSize(behavior.getSizeHint());
                                config.setMaxCachedCount(std::numeric_limits<size_t>::max());
                            }
                            result = CacheResource::make(baselineInstance(), config);
                        } else {
                            auto config = RegionResourceConfig::defaultConfig();
                            if (behavior.getSizeHint() != 0) {
                                config.setRegionSize(behavior.getSizeHint());
                                config.setAdaptive(false);
                            } else {
                                config.setAdaptive(true);
                            }
                            result = RegionResource::make(baselineInstance(), config);
                        }
                    } else if (behavior.repeating()) {
                        if (behavior.bulk() && behavior.getSizeHint() != 0) {
                            // allocate (hopefully) all required space up front
                            size_t sizeHintWithLeeway =
                                std::numeric_limits<size_t>::max() / 4 > behavior.getSizeHint()
                                    ? (behavior.getSizeHint() * 4) / 3
                                    : behavior.getSizeHint();
                            auto cacheConfig = CacheResourceConfig::defaultConfig();
                            // releasing memory back to the region is mostly pointless
                            cacheConfig.setMaxCachedCount(std::numeric_limits<size_t>::max());
                            cacheConfig.setMaxCacheSize(std::numeric_limits<size_t>::max());

                            auto regionConfig = RegionResourceConfig::defaultConfig();
                            regionConfig.setRegionSize(sizeHintWithLeeway);
                            regionConfig.setAdaptive(true);
                            result = CacheResource::make(
                                RegionResource::make(baselineInstance(), regionConfig),
                                cacheConfig);
                        } else {
                            auto config = CacheResourceConfig::defaultConfig();
                            config.setMaxCachedCount(std::numeric_limits<size_t>::max());
                            config.setMaxCacheSize(std::numeric_limits<size_t>::max());
                            result = CacheResource::make(baselineInstance(), config);
                        }
                    } else {
                        auto config = PoolResourceConfig::defaultConfig();
                        config.setChunkSize(behavior.getSizeHint());
                        config.setMaxChunkSize(behavior.getSizeHint());
                        config.setMaxCachedChunks(1);
                        // config must be valid. If not using defaultConfig seems appropriate.
                        result = PoolResource::make(baselineInstance(), config);
                    }
                    HINT =
                        std::make_unique<std::variant<MemoryResource, AllocationBehavior>>(result);
                    return result;
                }},
            *HINT);
    }

    const AllocationBehavior AllocationBehavior::MIXED{0, false, false, false};
    const AllocationBehavior AllocationBehavior::REPEATING{0, true, false, false};
    const AllocationBehavior AllocationBehavior::ALLOC_THEN_FULL_RELEASE{0, false, true, false};
    const AllocationBehavior AllocationBehavior::INCREMENTAL{0, false, false, false};
    const AllocationBehavior AllocationBehavior::BULK{0, false, false, true};

    AllocationBehavior::AllocationBehavior(size_t sizeHint, bool repeating, bool allocFullRelease,
                                           bool bulk)
        : _sizeHint{sizeHint},
          _repeating{repeating},
          _allocFullRelease{allocFullRelease},
          _bulk{bulk}
    {
    }

    AllocationBehavior AllocationBehavior::sizeHint(size_t size)
    {
        AllocationBehavior ret;
        ret._sizeHint = size;
        return ret;
    }

    AllocationBehavior AllocationBehavior::operator|(const AllocationBehavior& other) const
    {
        AllocationBehavior ret;
        ret._sizeHint = std::max(this->_sizeHint, other._sizeHint);
        ret._flags = this->_flags | other._flags;
        return ret;
    }

    size_t AllocationBehavior::getSizeHint() const
    {
        return _sizeHint;
    }

    bool AllocationBehavior::mixed() const
    {
        return !_repeating && !_allocFullRelease;
    }

    bool AllocationBehavior::repeating() const
    {
        return _repeating;
    }

    bool AllocationBehavior::allocThenFullRelease() const
    {
        return _allocFullRelease;
    }

    bool AllocationBehavior::incremental() const
    {
        return !_bulk;
    }

    bool AllocationBehavior::bulk() const
    {
        return _bulk;
    }

} // namespace elsa::mr::hint