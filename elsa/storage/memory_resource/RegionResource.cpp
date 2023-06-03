#include "RegionResource.h"

#include "BitUtil.h"
#include "Util.h"

namespace elsa::mr
{
    RegionResource::RegionResource(const MemoryResource& upstream,
                                   const RegionResourceConfig& config)
        : _upstream{upstream}, _config{config}
    {
        _basePtr = _upstream->allocate(_config.regionSize, region_resource::BLOCK_GRANULARITY);
        _bumpPtr = _basePtr;
        _allocatedSize = 0;
        _endPtr = detail::voidPtrOffset(_basePtr, _config.regionSize);
    }

    RegionResource::~RegionResource()
    {
        _upstream->deallocate(_basePtr, _config.regionSize, region_resource::BLOCK_GRANULARITY);
    }

    MemoryResource RegionResource::make(const MemoryResource& upstream,
                                        const RegionResourceConfig& config)
    {
        return MemoryResource::MakeRef(new RegionResource(upstream, config));
    }

    void* RegionResource::allocate(size_t size, size_t alignment)
    {
        auto [_, sizeWithAlignment] =
            util::computeSizeWithAlignment(size, alignment, region_resource::BLOCK_GRANULARITY);

        size_t remainingSize =
            reinterpret_cast<uintptr_t>(_endPtr) - reinterpret_cast<uintptr_t>(_bumpPtr);

        if (sizeWithAlignment > remainingSize) {
            return _upstream->allocate(sizeWithAlignment, alignment);
        }

        _allocatedSize += sizeWithAlignment;
        void* ret = detail::alignUp(_bumpPtr, alignment);
        _bumpPtr = detail::voidPtrOffset(_bumpPtr, sizeWithAlignment);
        return ret;
    }

    void RegionResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        if (reinterpret_cast<uintptr_t>(_basePtr) <= reinterpret_cast<uintptr_t>(ptr)
            && reinterpret_cast<uintptr_t>(ptr) < reinterpret_cast<uintptr_t>(_endPtr)) {
            auto [_, sizeWithAlignment] =
                util::computeSizeWithAlignment(size, alignment, region_resource::BLOCK_GRANULARITY);
            _allocatedSize -= sizeWithAlignment;

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
} // namespace elsa::mr
