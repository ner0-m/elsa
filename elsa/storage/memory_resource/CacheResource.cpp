#include "CacheResource.h"
#include <limits>

namespace elsa::mr
{
    CacheResourceConfig::CacheResourceConfig(size_t maxCacheSize, size_t maxCachedCount)
        : maxCacheSize{maxCacheSize}, maxCachedCount{maxCachedCount}
    {
    }

    CacheResourceConfig CacheResourceConfig::defaultConfig()
    {
        return CacheResourceConfig(std::numeric_limits<size_t>::max(), 16);
    }

    CacheResourceConfig& CacheResourceConfig::setMaxCacheSize(size_t size)
    {
        maxCacheSize = size;
        return *this;
    }

    CacheResourceConfig& CacheResourceConfig::setMaxCachedCount(size_t count)
    {
        maxCachedCount = count;
        return *this;
    }

    void CacheResource::releaseCache()
    {
        for (auto& entry : _cache) {
            _upstream->deallocate(entry.ptr, entry.size, entry.alignment);
        }
        _cache.clear();
        _sizeToCacheElement.clear();
    }

    CacheResource::CacheResource(MemoryResource upstream, const CacheResourceConfig& config)
        : _upstream{upstream}, _config{config}, _sizeToCacheElement{config.maxCachedCount}
    {
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
            _cache.erase(mapIt->second);
            return ptr;
        }
    }

    void CacheResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
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
        if (_cache.size() >= _config.maxCachedCount || _cachedSize >= _config.maxCacheSize) {
            cache_resource::CacheElement& poppedElement = _cache.front();
            auto poppedIt = _sizeToCacheElement.find({poppedElement.size, poppedElement.alignment});
            // TODO: exception if poppedElement is not found, i.e. the invariants are violated
            while (&*poppedIt->second != &poppedElement)
                poppedIt++;
            _sizeToCacheElement.erase(poppedIt);
            _cachedSize -= poppedElement.size;
            _cache.pop_front();
        }
    }

    bool CacheResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        return _upstream->tryResize(ptr, size, alignment, newSize);
    }

    void CacheResource::copyMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->copyMemory(ptr, src, size);
    }

    void CacheResource::setMemory(void* ptr, const void* src, size_t stride, size_t count)
    {
        _upstream->setMemory(ptr, src, stride, count);
    }

    void CacheResource::moveMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->moveMemory(ptr, src, size);
    }
} // namespace elsa::mr
