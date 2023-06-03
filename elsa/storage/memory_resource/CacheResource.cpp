#include "CacheResource.h"

namespace elsa::mr
{
    void CacheResource::releaseCache()
    {
        for (auto& entry : _cache) {
            _upstream->deallocate(entry.ptr, entry.size, entry.alignment);
        }
        _cache.clear();
        _sizeToCacheElement.clear();
    }

    CacheResource::CacheResource(MemoryResource upstream, const CacheResourceConfig& config)
        : _upstream{upstream}, _config{config}
    {
        if (config.maxCachedCount != std::numeric_limits<size_t>::max()) {
            // The + 1 is necessary, because elements are evicted after inserting,
            // so the cache is briefly larger than the maxCachedCount.
            _sizeToCacheElement.reserve(config.maxCachedCount + 1);
        }
    }

    CacheResource::~CacheResource()
    {
        releaseCache();
    }

    MemoryResource CacheResource::make(MemoryResource upstream, const CacheResourceConfig& config)
    {
        return MemoryResource::MakeRef(new CacheResource(upstream, config));
    }

    void* CacheResource::allocate(size_t size, size_t alignment)
    {
        auto mapIt = _sizeToCacheElement.find({size, alignment});
        if (mapIt == _sizeToCacheElement.end()) {
            try {
                return _upstream->allocate(size, alignment);
            } catch (std::bad_alloc& e) {
                releaseCache();
                // try again after hopefully returning enough memory to the upstream allocator
                return _upstream->allocate(size, alignment);
            }
        } else {
            _sizeToCacheElement.erase(mapIt);
            void* ptr = mapIt->second->ptr;
            _cachedSize -= mapIt->second->size;
            _cache.erase(mapIt->second);
            return ptr;
        }
    }

    void CacheResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        if (size > _config.maxCacheSize) {
            _upstream->deallocate(ptr, size, alignment);
            return;
        }

        if (!ptr) {
            return;
        }

        try {
            _cache.push_back({ptr, size, alignment});
        } catch (std::bad_alloc& e) {
            _upstream->deallocate(ptr, size, alignment);
            return;
        }

        try {
            _sizeToCacheElement.insert({{size, alignment}, --_cache.end()});
        } catch (std::bad_alloc& e) {
            _cache.pop_back();
            _upstream->deallocate(ptr, size, alignment);
            return;
        }

        _cachedSize += size;
        while (_cache.size() > _config.maxCachedCount || _cachedSize > _config.maxCacheSize) {
            cache_resource::CacheElement& poppedElement = _cache.front();
            auto poppedIt = _sizeToCacheElement.find({poppedElement.size, poppedElement.alignment});
            // If this throws, internal invariants are violated (the element is not in the map).
            // This would be a serious bug, thus termination seems justified.
            while (&*poppedIt->second != &poppedElement)
                poppedIt++;
            _sizeToCacheElement.erase(poppedIt);
            _cachedSize -= poppedElement.size;
            _upstream->deallocate(poppedElement.ptr, poppedElement.size, poppedElement.alignment);
            _cache.pop_front();
        }
    }

    bool CacheResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        return _upstream->tryResize(ptr, size, alignment, newSize);
    }
} // namespace elsa::mr
