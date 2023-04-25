#include "doctest/doctest.h"

#include "memory_resource/ContiguousStorage.h"
#include "memory_resource/PrimitiveResource.h"
#include "Assertions.h"

using namespace elsa::mr;

class CountedResource : private PrimitiveResource
{
private:
    size_t _count = 0;

private:
    void* allocate(size_t size, size_t alignment) override
    {
        void* p = PrimitiveResource::allocate(size, alignment);
        _count += size;
        return p;
    }
    void deallocate(void* ptr, size_t size, size_t alignment) override
    {
        _count -= size;
        PrimitiveResource::deallocate(ptr, size, alignment);
    }
    bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override
    {
        if (!PrimitiveResource::tryResize(ptr, size, alignment, newSize))
            return false;
        _count += (newSize - size);
        return true;
    }

public:
    static MemoryResource make() { return MemoryResource::MakeRef(new CountedResource()); }

    size_t retrieve()
    {
        size_t count = _count;
        _count = 0;
        return count;
    }
};

TEST_SUITE_BEGIN("contiguousstorage");

TEST_CASE("Constructions")
{
    MemoryResource mres = CountedResource::make();

    GIVEN("Default constructor") {}
}
