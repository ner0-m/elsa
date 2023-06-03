#pragma once

#include <thrust/mr/memory_resource.h>
#include <thrust/universal_ptr.h>
#include "memory_resource/ContiguousMemory.h"

namespace elsa::mr
{
    class ElsaThrustMRAdaptor final
        : public thrust::mr::memory_resource<thrust::universal_ptr<void>>
    {
    private:
        MemoryResource _mr;

    public:
        using thrust::mr::memory_resource<thrust::universal_ptr<void>>::pointer;
        pointer do_allocate(std::size_t bytes, std::size_t alignment) override;
        void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) override;
        __host__ __device__ bool do_is_equal(const memory_resource& other) const noexcept override;

    public:
        ElsaThrustMRAdaptor();
    };
} // namespace elsa::mr