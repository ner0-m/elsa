#include "doctest/doctest.h"

#include "memory_resource/ContiguousMemory.h"
#include "memory_resource/UniversalResource.h"
#include "memory_resource/PoolResource.h"

TEST_SUITE_BEGIN("memoryresources");

TEST_CASE_TEMPLATE("Pool resource", T, float, double)
{
    GIVEN("An zero sized container")
    {
        using namespace elsa::mr;
        UniversalResource* upstream = new UniversalResource();
        PoolResource* pool = new PoolResource(upstream);
        pool->releaseRef();
        CHECK_EQ(4, 5);
    }
}

TEST_SUITE_END();
