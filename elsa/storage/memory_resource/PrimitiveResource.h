#pragma once

#include "ContiguousMemory.h"

namespace elsa::mr
{
    class PrimitiveResource : public MemResInterface
    {
    public:
        static MemoryResource make();

    public:
        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                       size_t newAlignment) override;
        void copyMemory(void* ptr, const void* src, size_t size) noexcept override;
        void moveMemory(void* ptr, const void* src, size_t size) noexcept override;
        void setMemory(void* ptr, const void* src, size_t stride, size_t count) noexcept override;
    };
} // namespace elsa::mr
