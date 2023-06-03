#pragma once

#include <algorithm>
#include <memory>
#include <variant>
#include <optional>
#include "ContiguousMemory.h"

namespace elsa::mr::hint
{
    class AllocationBehavior
    {
        friend struct behavior;

    private:
        size_t _sizeHint{0};
        union {
            char _flags{0};

            struct {
                char _repeating : 1;
                char _allocFullRelease : 1;
                char _bulk : 1;
            };
        };

        constexpr AllocationBehavior(size_t sizeHint, bool repeating, bool allocFullRelease,
                                     bool bulk)
            : _sizeHint{sizeHint},
              _repeating{repeating},
              _allocFullRelease{allocFullRelease},
              _bulk{bulk}
        {
        }
        constexpr AllocationBehavior() : _sizeHint{0}, _flags{0} {}

    public:
        /// @brief Hint about the maximum amount of memory that may be allocated from the resource
        /// at any given time.
        static AllocationBehavior sizeHint(size_t size)
        {
            AllocationBehavior ret;
            ret._sizeHint = size;
            return ret;
        }

        constexpr AllocationBehavior operator|(const AllocationBehavior& other) const
        {
            AllocationBehavior ret;
            ret._sizeHint = std::max(this->_sizeHint, other._sizeHint);
            ret._flags = this->_flags | other._flags;
            return ret;
        }
        constexpr size_t getSizeHint() const { return _sizeHint; }
        constexpr bool mixed() const { return !_repeating && !_allocFullRelease; }
        constexpr bool repeating() const { return _repeating; }
        constexpr bool allocThenFullRelease() const { return _allocFullRelease; }
        constexpr bool incremental() const { return !_bulk; }
        constexpr bool bulk() const { return _bulk; }
    };

    struct behavior {
        /// @brief  Allocations and deallocations do not follow a pattern. Few to no assumptions can
        /// be made about allocation behaviour. Do not specify together with REPEATING or
        /// ALLOC_THEN_FULL_RELEASE.
        static constexpr AllocationBehavior MIXED{0, false, false, false};
        /// @brief Ideal for an repeated allocation and deallocation of blocks of the same size and
        /// alignment, e.g. in a loop.
        static constexpr AllocationBehavior REPEATING{0, true, false, false};
        /// @brief Ideal for a triangle allocation pattern, i.e. first allocate, potentially through
        /// multiple allocations, then release. Only specify this behavior when all the memory is
        /// released after each iteration, otherwise the released memory may not be reused
        static constexpr AllocationBehavior ALLOC_THEN_FULL_RELEASE{0, false, true, false};
        /// @brief Many allocations small are made. Do not specify together with BULK.
        static constexpr AllocationBehavior INCREMENTAL{0, false, false, false};
        /// @brief Few large allocations are made.
        static constexpr AllocationBehavior BULK{0, false, false, true};
    };

    namespace detail
    {
        using AllocationHint = std::unique_ptr<std::variant<MemoryResource, AllocationBehavior>>;
    }

    /// @brief Provides hints that help determine an appropriate memory resource. The hints are
    /// limited to the scope in which they are created and anything called from there. Hints are
    /// meant to be stack allocated. DO NOT HEAP ALLOCATE THEM OR STORE THEM IN MEMBERS!
    /// Note: each hint is responsible for the creation of up to 1 memory resource, meaning the
    /// resource is shared by subsequent calls to elsa::mr::defaultInstance().
    /// @tparam H The type of the given hint. Two variants are accepted:
    /// elsa::mr::hint::AllocationBehavior and elsa::mr::MemoryResource
    template <typename H>
    class ScopedAllocationHint
    {
    private:
        detail::AllocationHint _previous;

    public:
        ScopedAllocationHint(const H& hint);
        ~ScopedAllocationHint();

        void* operator new(size_t) = delete;
        void* operator new(size_t, size_t) = delete;
        void operator delete(void*) = delete;
        void operator delete(void*, size_t) = delete;
    };

    /// DO NOT HEAP ALLOCATE OR STORE IN A MEMBER!
    using ScopedMR = ScopedAllocationHint<MemoryResource>;
    /// DO NOT HEAP ALLOCATE OR STORE IN A MEMBER!
    using ScopedMRHint = ScopedAllocationHint<AllocationBehavior>;
    /// @brief Select a memory resource based on the current hint. The preferred way to obtain a
    /// MemoryResource is to call elsa::mr::defaultInstance().
    /// @return std::nullopt if there is no hint in the current scope. An appropriate resource
    /// otherwise.
    std::optional<MemoryResource> selectMemoryResource();
} // namespace elsa::mr::hint