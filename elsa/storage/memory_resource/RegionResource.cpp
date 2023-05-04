#include "RegionResource.h"

#include "BitUtil.h"

namespace elsa::mr
{
    RegionResourceConfig::RegionResourceConfig(size_t regionSize, bool adaptive)
        : regionSize{regionSize}, isAdaptive{adaptive}
    {
    }

    RegionResourceConfig RegionResourceConfig::defaultConfig()
    {
        return RegionResourceConfig(31, true);
    }

    RegionResourceConfig& RegionResourceConfig::setAdaptive(bool adaptive)
    {
        isAdaptive = adaptive;
        return *this;
    }

    RegionResourceConfig& RegionResourceConfig::setRegionSize(size_t size)
    {
        regionSize = size;
        return *this;
    }

    RegionResource::RegionResource(const MemoryResource& upstream,
                                   const RegionResourceConfig& config)
        : _upstream{upstream}, _config{config}
    {
        _basePtr = _upstream->allocate(_config.regionSize, 8);
        _bumpPtr = _basePtr;
        _allocatedSize = 0;
        _endPtr = voidPtrOffset(_basePtr, _config.regionSize);
    }

    MemoryResource RegionResource::make(const MemoryResource& upstream,
                                        const RegionResourceConfig& config)
    {
        return MemoryResource::MakeRef(new RegionResource(upstream, config));
    }

    void* RegionResource::allocate(size_t size, size_t alignment)
    {
        size_t remainingSize =
            reinterpret_cast<uintptr_t>(_endPtr) - reinterpret_cast<uintptr_t>(_bumpPtr);

        if (size > remainingSize) {
            return _upstream->allocate(size, alignment);
        }

        _allocatedSize += size;
        void* ret = _bumpPtr;
        _bumpPtr = voidPtrOffset(_bumpPtr, size);
        return ret;
    }

    void RegionResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        if (reinterpret_cast<uintptr_t>(_basePtr) <= reinterpret_cast<uintptr_t>(ptr)
            && reinterpret_cast<uintptr_t>(ptr) < reinterpret_cast<uintptr_t>(_endPtr)) {
            _allocatedSize -= size;

            if (_allocatedSize == 0) {
                // arena can be reused
                _bumpPtr = _basePtr;
                // TODO: if config is set to adaptive, potentially allocate new arena based on
                // observed allocation behavior
            }
        } else {
            _upstream->deallocate(ptr, size, alignment);
            return;
        }
    }

    bool RegionResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        if (reinterpret_cast<uintptr_t>(_basePtr) <= reinterpret_cast<uintptr_t>(ptr)
            && reinterpret_cast<uintptr_t>(ptr) < reinterpret_cast<uintptr_t>(_endPtr)) {
            return false;
        } else {
            return _upstream->tryResize(ptr, size, alignment, newSize);
        }
    }

    void RegionResource::copyMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->copyMemory(ptr, src, size);
    }

    void RegionResource::setMemory(void* ptr, const void* src, size_t stride, size_t count)
    {
        _upstream->setMemory(ptr, src, stride, count);
    }

    void RegionResource::moveMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->moveMemory(ptr, src, size);
    }
} // namespace elsa::mr