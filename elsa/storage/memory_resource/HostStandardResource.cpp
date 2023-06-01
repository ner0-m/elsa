#include "HostStandardResource.h"

#include "BitUtil.h"
#include "Util.h"

#include <memory>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace elsa::mr
{
    MemoryResource HostStandardResource::make()
    {
        return MemoryResource::MakeRef(new HostStandardResource());
    }

    void* HostStandardResource::allocate(size_t size, size_t alignment)
    {
        if (size == 0) {
            ++size;
        }
        if (!isPowerOfTwo(alignment)) {
            throw std::bad_alloc();
        }

        // std::aligned_alloc requires size as a multiple of alignment
        size = alignUp(size, alignment);
        void* ptr = std::aligned_alloc(alignment, size);
        if (unlikely(!ptr)) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void HostStandardResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        std::free(ptr);
    }

    bool HostStandardResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        static_cast<void>(ptr);
        static_cast<void>(size);
        static_cast<void>(alignment);
        static_cast<void>(newSize);
        return false;
    }
} // namespace elsa::mr
