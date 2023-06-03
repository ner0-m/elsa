#pragma once

#include "MemoryResource.h"
#include <unordered_map>
#include <list>
#include <memory>
#include <limits>

template <>
class std::hash<std::pair<size_t, size_t>>
{
public:
    size_t operator()(const std::pair<size_t, size_t>& pair) const
    {
        return std::hash<size_t>()(std::hash<size_t>()(pair.first) ^ pair.second);
    }
};

namespace elsa::mr
{
    namespace cache_resource
    {
        struct CacheElement {
            void* ptr;

            size_t size;

            size_t alignment;
        };
    } // namespace cache_resource

    class CacheResource;

    class CacheResourceConfig
    {
    private:
        friend class CacheResource;

        size_t maxCacheSize;

        size_t maxCachedCount;

        constexpr CacheResourceConfig(size_t maxCacheSize, size_t maxCachedCount)
            : maxCacheSize{maxCacheSize}, maxCachedCount{maxCachedCount}
        {
        }

    public:
        /// @brief Default configuration for a cache resource with (hopefully) sensible defaults.
        /// @return Default configuration for a cache resource.
        static constexpr CacheResourceConfig defaultConfig()
        {
            return CacheResourceConfig(std::numeric_limits<size_t>::max(), 16);
        }

        /// @brief Set the maximum cumulative size of cached chunks, before releasing chunks to the
        /// upstream allocator
        /// @param size Maximum cumulative size of cached chunks
        /// @return self
        constexpr CacheResourceConfig& setMaxCacheSize(size_t size)
        {
            maxCacheSize = size;
            return *this;
        }

        /// @brief Set the maximum number of cached chunks, before releasing chunks to the
        /// upstream allocator
        /// @param count Maximum number of cached chunks. For an unlimited number of chunks,
        /// set this value to std::numeric_limits<usize>::max(). Be aware that, for any other
        /// value, space for the cache entries may be pre-reserved.
        /// @return self
        constexpr CacheResourceConfig& setMaxCachedCount(size_t count)
        {
            maxCachedCount = count;
            return *this;
        }
    };

    /// @brief Memory resource for the ContiguousStorage class.
    /// It caches freed blocks before returning them to the upstream allocator. Using this
    /// resource makes sense when the allocation pattern repeatedly allocates and frees blocks of
    /// exactly the same size, e.g. in a loop.
    /// IMPORTANT: THIS RESOURCE IS NOT SYNCHRONIZED!
    /// Advantage over a plain UniversalResource: allocations/deallocations are faster, memory is
    /// potentially already mapped from previous use. Only benefitial for repeating allocation
    /// patterns. Disadvantage: Move assignment between containers with different memory resources
    /// is more costly.
    class CacheResource : public MemResInterface
    {
    private:
        using Cache = std::list<cache_resource::CacheElement>;
        MemoryResource _upstream;

        CacheResourceConfig _config;

        std::unordered_multimap<std::pair<size_t, size_t>, Cache::iterator> _sizeToCacheElement;

        Cache _cache;

        size_t _cachedSize{0};

        void releaseCache();

    public:
        CacheResource(const CacheResource& other) = delete;

        CacheResource& operator=(const CacheResource& other) = delete;

        CacheResource(CacheResource&& other) noexcept = delete;

        CacheResource& operator=(CacheResource&& other) noexcept = delete;

    protected:
        CacheResource(MemoryResource upstream,
                      const CacheResourceConfig& config = CacheResourceConfig::defaultConfig());

        ~CacheResource();

    public:
        static MemoryResource
            make(MemoryResource upstream = globalResource(),
                 const CacheResourceConfig& config = CacheResourceConfig::defaultConfig());

        void* allocate(size_t size, size_t alignment) override;

        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;

        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) noexcept override;
    };
} // namespace elsa::mr
