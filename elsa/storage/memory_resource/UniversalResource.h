#pragma once
#include "ContiguousMemory.h"

// TODO: find a way to determine the presence of cuda
#if 0
namespace elsa::mr
{
    class UniversalResource : public MemResInterface
    {
    public:
        static MemoryResource make();

    public:
        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
        void copyMemory(void* ptr, const void* src, size_t size) override;
        void moveMemory(void* ptr, const void* src, size_t size) override;
        void setMemory(void* ptr, const void* src, size_t stride, size_t count) override;
    };
} // namespace elsa::mr
#endif