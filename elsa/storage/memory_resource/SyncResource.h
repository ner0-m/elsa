#pragma once

#include "ContiguousMemory.h"
#include <mutex>

namespace elsa::mr
{
    /// @brief Trivially synchs all allocations and deallocations to the wrapped resource with a
    /// single lock.
    template <typename T>
    class SyncResource : public T
    {
    private:
        std::mutex _m;

    protected:
        template <typename... Ts>
        SyncResource(Ts&&... args);

    public:
        /// @brief Creates a SyncResource wrapping a back-end resource, which performs the actual
        /// allocations.
        /// @param ...args Parameters passed to the constructor of the wrapped resource.
        /// @return A MemoryResource encapsulationg the SynchedResource
        template <typename... Ts>
        static MemoryResource make(Ts&&... args);

        /// @brief Allocates from the wrapped resource. Blocking until the resource is not busy.
        void* allocate(size_t size, size_t alignment) override;
        /// @brief Passes the pointer along to the wrapped resource for deallocation. Blocking until
        /// the resource is not busy.
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
        /// @brief Tries to have the wrapped resource resize the allocation. Blocking until the
        /// resource is not busy.
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
    };

    template <typename T>
    inline void* SyncResource<T>::allocate(size_t size, size_t alignment)
    {
        std::unique_lock lock{_m};
        return T::allocate(size, alignment);
    }

    template <typename T>
    inline void SyncResource<T>::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        std::unique_lock lock{_m};
        return T::deallocate(ptr, size, alignment);
    }

    template <typename T>
    inline bool SyncResource<T>::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        std::unique_lock lock{_m};
        return T::tryResize(ptr, size, alignment, newSize);
    }

    template <typename T>
    template <typename... Ts>
    inline SyncResource<T>::SyncResource(Ts&&... args) : T{std::forward<Ts>(args)...}
    {
    }

    template <typename T>
    template <typename... Ts>
    inline MemoryResource SyncResource<T>::make(Ts&&... args)
    {
        return MemoryResource::MakeRef(new SyncResource<T>(std::forward<Ts>(args)...));
    }
} // namespace elsa::mr