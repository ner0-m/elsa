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

    using ScopedMR = ScopedAllocationHint<MemoryResource>;
    using ScopedMRHint = ScopedAllocationHint<AllocationBehavior>;

    std::optional<MemoryResource> selectMemoryResource();
} // namespace elsa::mr::hint