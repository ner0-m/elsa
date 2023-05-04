#include <memory>
#include <variant>
#include <optional>
#include "ContiguousMemory.h"

namespace elsa::mr::hint
{
    class AllocationBehavior
    {
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

        AllocationBehavior(size_t sizeHint, bool repeating, bool allocFullRelease, bool bulk);
        AllocationBehavior() = default;

    public:
        /// @brief  Allocations and deallocations do not follow a pattern. Few to no assumptions can
        /// be made about allocation behaviour. Do not specify together with REPEATING or
        /// ALLOC_THEN_FULL_RELEASE.
        static const AllocationBehavior MIXED;
        /// @brief Ideal for an repeated allocation and deallocation of blocks of the same size and
        /// alignment, e.g. in a loop.
        static const AllocationBehavior REPEATING;
        /// @brief Ideal for a triangle allocation pattern, i.e. first allocate, potentially through
        /// multiple allocations, then release. Only specify this behavior when all the memory is
        /// released after each iteration, otherwise the released memory may not be reused!
        static const AllocationBehavior ALLOC_THEN_FULL_RELEASE;
        /// @brief Many allocations small are made. Do not specify together with BULK.
        static const AllocationBehavior INCREMENTAL;
        /// @brief Few large allocations are made.
        static const AllocationBehavior BULK;

        /// @brief Hint about the maximum amount of memory that may be allocated from the resource
        /// at any given time.
        static AllocationBehavior sizeHint(size_t size);

        AllocationBehavior operator|(const AllocationBehavior& other) const;
        size_t getSizeHint() const;
        bool mixed() const;
        bool repeating() const;
        bool allocThenFullRelease() const;
        bool incremental() const;
        bool bulk() const;
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