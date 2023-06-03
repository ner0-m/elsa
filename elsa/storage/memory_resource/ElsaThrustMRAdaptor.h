#pragma once

#include <thrust/mr/memory_resource.h>
#include <thrust/universal_ptr.h>
#include "memory_resource/ContiguousMemory.h"

namespace elsa::mr
{
    /// @brief Wraps an elsa::mr::MemoryResource into a thrust::mr::memory_resource for use with
    /// e.g. thrust::universal_vector.
    ///
    /// thrust::mr::memory_resource::do_allocate -> elsa::mr::MemResInterface::allocate
    ///
    /// thrust::mr::memory_resource::do_deallocate -> elsa::mr::MemResInterface::deallocate
    ///
    /// thrust::mr::memory_resource::do_is_equal -> basic address comparison with other,
    /// cannot be mapped directly to elsa::mr::MemoryResource::operator==, because it must be
    /// tagged as __host__ __device__
    class ElsaThrustMRAdaptor final
        : public thrust::mr::memory_resource<thrust::universal_ptr<void>>
    {
    private:
        MemoryResource _mr;

    public:
        using thrust::mr::memory_resource<thrust::universal_ptr<void>>::pointer;
        /// @brief maps to elsa::mr::MemResInterface::allocate
        pointer do_allocate(std::size_t bytes, std::size_t alignment) override;
        /// @brief maps to elsa::mr::MemResInterface::deallocate
        void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) override;
        /// @return true iff. this == &other
        __host__ __device__ bool do_is_equal(const memory_resource& other) const noexcept override;

    public:
        ElsaThrustMRAdaptor();
    };
} // namespace elsa::mr