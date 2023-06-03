#pragma once

#include "ContiguousMemory.h"

namespace elsa::mr
{
    /// @brief Wraps a thrust::mr::memory_resource into the elsa::mr::MemResInterface for use with
    /// e.g. elsa::mr::ContiguousVector
    /// @tparam T type of the wrapped thrust resource
    /// elsa::mr::MemResInterface::allocate -> thrust::mr::memory_resource::do_allocate
    ///
    /// elsa::mr::MemResInterface::deallocate -> thrust::mr::memory_resource::do_deallocate
    ///
    /// elsa::mr::MemResInterface::tryResize -> false
    template <typename T>
    class ThrustElsaMRAdaptor : public MemResInterface
    {
    private:
        T _thrustMR;

    protected:
        template <typename... Ts>
        ThrustElsaMRAdaptor(Ts&&... args) : _thrustMR{std::forward<Ts>(args)...} {};

    public:
        /// @brief maps to thrust::mr::memory_resource::do_allocate
        void* allocate(size_t size, size_t alignment) override
        {
            return _thrustMR.do_allocate(size, alignment).get();
        };
        /// @brief maps to thrust::mr::memory_resource::do_deallocate
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
        {
            return _thrustMR.do_deallocate((typename T::pointer)(ptr), size, alignment);
        }
        /// @brief This is a no-op.
        /// @return false
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override
        {
            return false;
        }

        template <typename... Ts>
        static MemoryResource make(Ts&&... args)
        {
            return MemoryResource::MakeRef(new ThrustElsaMRAdaptor<T>(std::forward<Ts>(args)...));
        };
    };
} // namespace elsa::mr
