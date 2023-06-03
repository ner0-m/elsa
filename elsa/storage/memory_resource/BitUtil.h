#pragma once

#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <limits>

namespace elsa::mr::detail
{
    // NOLINTBEGIN
    template <typename T>
    static constexpr T bit_width(T t)
    {
        // TODO: use std::bit_width when C++20 becomes available, or find a useful bit hack for this
        T c = 0;
        while (t) {
            t >>= 1;
            ++c;
        }
        return c;
    }

    template <typename T>
    static constexpr T log2Floor(T t)
    {
        return bit_width(t) - 1;
    }

    // does not work for t == 0
    template <typename T>
    static constexpr T log2Ceil(T t)
    {
        return bit_width(t - 1);
    }

    // not zero indexed! lowestSetBit(0) == 0!
    template <typename T>
    static constexpr T lowestSetBit(T t)
    {
        return bit_width(t & ~(t - 1));
    }

    template <typename T>
    static constexpr bool isPowerOfTwo(T t)
    {
        return (t != 0) && !(t & (t - 1));
    }

    // alignment must be a power of 2
    template <typename T>
    static constexpr T alignDown(T value, T alignment)
    {
        return value & ~(alignment - 1);
    }

    // alignment must be a power of 2
    template <typename T>
    static constexpr T alignUp(T value, T alignment)
    {
        return alignDown(value + alignment - 1, alignment);
    }

    // alignment must be a power of 2
    static inline void* alignDown(void* ptr, size_t alignment)
    {
        return reinterpret_cast<void*>(alignDown(reinterpret_cast<uintptr_t>(ptr), alignment));
    }

    // alignment must be a power of 2
    static inline void* alignUp(void* ptr, size_t alignment)
    {
        return reinterpret_cast<void*>(alignUp(reinterpret_cast<uintptr_t>(ptr), alignment));
    }

    // alignment must be a power of 2
    static inline bool checkAlignment(void* ptr, size_t alignment)
    {
        return ptr == alignDown(ptr, alignment);
    }

    static inline void* voidPtrOffset(void* ptr, size_t offset)
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + offset);
    }
    // NOLINTEND
} // namespace elsa::mr::detail