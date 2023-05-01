#include "HostStandardResource.h"
#include "BitUtil.h"

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
        if (!ptr) [[unlikely]] {
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

    void HostStandardResource::copyMemory(void* ptr, const void* src, size_t size)
    {
        std::memcpy(ptr, src, size);
    }
    void HostStandardResource::moveMemory(void* ptr, const void* src, size_t size)
    {
        std::memmove(ptr, src, size);
    }

    namespace detail
    {
        template <class Type>
        void typedFill(Type* ptr, const Type* src, size_t stride, size_t totalWrites)
        {
            const Type* s = src;
            const Type* e = src + stride;

            for (size_t i = 0; i < totalWrites; i++) {
                *ptr = *s;
                ++ptr;
                if (++s == e)
                    s = src;
            }
        }

        template <class Type>
        void pointerFill(void* ptr, const void* src, size_t stride, size_t count)
        {
            typedFill<Type>(static_cast<Type*>(ptr), static_cast<const Type*>(src), stride,
                            stride * count);
        }
    } // namespace detail

    void HostStandardResource::setMemory(void* ptr, const void* src, size_t stride, size_t count)
    {
        if ((stride % 8) == 0)
            detail::pointerFill<uint64_t>(ptr, src, (stride / 8), count);
        else if ((stride % 4) == 0)
            detail::pointerFill<uint32_t>(ptr, src, (stride / 4), count);
        else if ((stride % 2) == 0)
            detail::pointerFill<uint16_t>(ptr, src, (stride / 2), count);
        else
            detail::pointerFill<uint8_t>(ptr, src, stride, count);
    }
} // namespace elsa::mr
