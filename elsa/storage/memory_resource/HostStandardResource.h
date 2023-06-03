#pragma once

#include "MemoryResource.h"

namespace elsa::mr
{
    class HostStandardResource : public MemResInterface
    {
    protected:
        HostStandardResource() = default;

    public:
        static MemoryResource make();

    public:
        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) noexcept override;
    };
} // namespace elsa::mr
