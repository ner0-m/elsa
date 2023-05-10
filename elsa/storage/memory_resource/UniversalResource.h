#pragma once
#include "ContiguousMemory.h"

#ifdef ELSA_CUDA_ENABLED
namespace elsa::mr
{
    class UniversalResource : public MemResInterface
    {
    protected:
        UniversalResource() = default;

    public:
        static MemoryResource make();

    public:
        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
    };
} // namespace elsa::mr
#else
#include "HostStandardResource.h"

namespace elsa::mr
{
    using UniversalResource = HostStandardResource;
}
#endif