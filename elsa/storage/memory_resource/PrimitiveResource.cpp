#include "PrimitiveResource.h"

#include "BitUtil.h"
#include <memory>
#include <cstdlib>
#include <cstring>

namespace elsa::mr
{
    MemoryResource PrimitiveResource::make()
    {
        return MemoryResource::MakeRef(new PrimitiveResource());
    }

    void* PrimitiveResource::allocate(size_t size, size_t alignment)
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
        if (!ptr) [[unlikely]] {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void PrimitiveResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        std::free(ptr);
    }

    bool PrimitiveResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                                      size_t newAlignment)
    {
        static_cast<void>(ptr);
        static_cast<void>(size);
        static_cast<void>(alignment);
        static_cast<void>(newSize);
        static_cast<void>(newAlignment);
        return false;
    }

    void PrimitiveResource::copyMemory(void* ptr, const void* src, size_t size) noexcept
    {
        std::memcpy(ptr, src, size);
    }
    void PrimitiveResource::moveMemory(void* ptr, const void* src, size_t size) noexcept
    {
        std::memmove(ptr, src, size);
    }

    namespace detail
    {
        template <class Type>
        void typedFill(void* ptr, const void* src, size_t count)
        {
            std::fill(static_cast<Type*>(ptr), static_cast<Type*>(ptr) + count,
                      *static_cast<const Type*>(src));
        }
    } // namespace detail

    void PrimitiveResource::setMemory(void* ptr, const void* src, size_t stride,
                                      size_t count) noexcept
    {
        if ((stride % 8) == 0)
            detail::typedFill<uint64_t>(ptr, src, count * (stride / 8));
        else if ((stride % 4) == 0)
            detail::typedFill<uint32_t>(ptr, src, count * (stride / 4));
        else if ((stride % 2) == 0)
            detail::typedFill<uint16_t>(ptr, src, count * (stride / 2));
        else
            detail::typedFill<uint8_t>(ptr, src, count * stride);
    }
} // namespace elsa::mr
