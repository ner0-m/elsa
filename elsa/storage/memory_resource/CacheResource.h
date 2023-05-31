#pragma once
#include "ContiguousMemory.h"
#include <unordered_map>
#include <list>
#include <memory>

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

        CacheResourceConfig(size_t maxCacheSize, size_t maxCachedCount);

    public:
        /// @brief Default configuration for a cache resource with (hopefully) sensible defaults.
        /// @return Default configuration for a cache resource.
        static CacheResourceConfig defaultConfig();

        /// @brief Set the maximum cumulative size of cached chunks, before releasing chunks to the
        /// upstream allocator
        /// @param size Maximum cumulative size of cached chunks
        /// @return self
        CacheResourceConfig& setMaxCacheSize(size_t size);

        /// @brief Set the maximum number of cached chunks, before releasing chunks to the
        /// upstream allocator
        /// @param count Maximum number of cached chunks. For an unlimited number of chunks,
        /// set this value to std::numeric_limits<usize>::max(). Be aware that, for any other
        /// value, space for the cache entries may be pre-reserved.
        /// @return self
        CacheResourceConfig& setMaxCachedCount(size_t count);
    };

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
            make(MemoryResource upstream = baselineInstance(),
                 const CacheResourceConfig& config = CacheResourceConfig::defaultConfig());

        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
    };
} // namespace elsa::mr
