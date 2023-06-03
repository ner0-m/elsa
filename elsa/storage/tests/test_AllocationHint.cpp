#include "doctest/doctest.h"

#include "memory_resource/ContiguousMemory.h"
#include "memory_resource/AllocationHint.h"

namespace elsa::mr
{
    class DummyResource : public MemResInterface
    {
    protected:
        DummyResource() {}
        ~DummyResource() = default;

    public:
        static MemoryResource make() { return MemoryResource::MakeRef(new DummyResource()); }

        void* allocate(size_t size, size_t alignment) override
        {
            static_cast<void>(size);
            static_cast<void>(alignment);
            throw std::bad_alloc{};
        }
        void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
        {
            static_cast<void>(size);
            static_cast<void>(ptr);
            static_cast<void>(alignment);
        }
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override
        {
            static_cast<void>(ptr);
            static_cast<void>(size);
            static_cast<void>(alignment);
            static_cast<void>(newSize);
            return false;
        }
    };
} // namespace elsa::mr

TEST_SUITE_BEGIN("allocationhints");

TEST_CASE("Memory resource hint")
{
    GIVEN("No hint")
    {
        using namespace elsa::mr;
        CHECK_EQ(defaultInstance(), baselineInstance());
    }

    GIVEN("Scoped memory resource hint")
    {
        using namespace elsa::mr;
        MemoryResource dummyResource = DummyResource::make();
        {
            hint::ScopedMR hint{dummyResource};
            auto tmp = defaultInstance();
            auto tmp2 = baselineInstance();
            CHECK_NE(defaultInstance(), baselineInstance());
            CHECK_EQ(defaultInstance(), dummyResource);
            {
                hint::ScopedMR hint{DummyResource::make()};
                CHECK_NE(defaultInstance(), dummyResource);
            }
            CHECK_EQ(defaultInstance(), dummyResource);
            hint::ScopedMR hint2{DummyResource::make()};
            CHECK_NE(defaultInstance(), dummyResource);
        }
        CHECK_EQ(defaultInstance(), baselineInstance());
    }
}

TEST_CASE("Allocation pattern hint")
{
    GIVEN("Mixed hints")
    {
        using namespace elsa::mr;

        auto hint = hint::behavior::ALLOC_THEN_FULL_RELEASE | hint::behavior::BULK
                    | hint::AllocationBehavior::sizeHint(0x1000);
        CHECK(hint.allocThenFullRelease());
        CHECK(hint.bulk());
        CHECK_EQ(hint.getSizeHint(), 0x1000);
        CHECK_FALSE(hint.mixed());
        CHECK_FALSE(hint.repeating());
        CHECK_FALSE(hint.incremental());
    }
}

TEST_SUITE_END;