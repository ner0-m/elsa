#pragma once
#include <cstring>
#include <atomic>

namespace elsa::mr
{
    class MemoryResource
    {
    private:
        std::atomic<size_t> _refCount;

    public:
        MemoryResource();
        virtual ~MemoryResource() = default;

        virtual void* allocate(size_t size, size_t alignment) = 0;
        virtual bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                               size_t newAlignment) = 0;
        virtual void deallocate(void* ptr, size_t size, size_t alignment) = 0;

        virtual void copyMemory(void* ptr, const void* src, size_t size) noexcept = 0;
        virtual void setMemory(void* ptr, const void* src, size_t stride,
                               size_t count) noexcept = 0;
        virtual void moveMemory(void* ptr, const void* src, size_t size) noexcept = 0;

        void addRef();
        void releaseRef();
    };

    MemoryResource* defaultInstance();
} // namespace elsa::mr