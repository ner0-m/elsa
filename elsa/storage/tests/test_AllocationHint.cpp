#include "doctest/doctest.h"

#include <iostream>

#include "memory_resource/MemoryResource.h"
#include "memory_resource/AllocationHint.h"

namespace elsa::mr
{
    class DummyResource : public MemResInterface
    {
    protected:
        DummyResource() {}
        ~DummyResource() = default;

    public:
        static MemoryResource make()
        {
            return std::shared_ptr<MemResInterface>(new DummyResource(),
                                                    [](DummyResource* p) { delete p; });
        }

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
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) noexcept override
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
        CHECK_EQ(defaultResource(), globalResource());
    }

    GIVEN("Scoped memory resource hint")
    {
        using namespace elsa::mr;
        MemoryResource dummyResource = DummyResource::make();
        {
            hint::ScopedMR hint{dummyResource};
            auto tmp = defaultResource();
            auto tmp2 = globalResource();
            CHECK_NE(defaultResource(), globalResource());
            CHECK_EQ(defaultResource(), dummyResource);
            {
                hint::ScopedMR hint{DummyResource::make()};
                CHECK_NE(defaultResource(), dummyResource);
            }
            CHECK_EQ(defaultResource(), dummyResource);
            hint::ScopedMR hint2{DummyResource::make()};
            CHECK_NE(defaultResource(), dummyResource);
        }
        CHECK_EQ(defaultResource(), globalResource());
    }
}

TEST_SUITE_END;