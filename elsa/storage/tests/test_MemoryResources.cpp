#include "doctest/doctest.h"

#include "memory_resource/ContiguousMemory.h"
#include "memory_resource/UniversalResource.h"

#include <cstring>

TEST_SUITE_BEGIN("memoryresources");

TEST_CASE_TEMPLATE("Universal resource", T, float, double)
{
    GIVEN("Check overlap")
    {
        using namespace elsa::mr;
        UniversalResource* univ = new UniversalResource();
        unsigned char* ptrs[100];
        for (int i = 0; i < 100; i++) {
            ptrs[i] = reinterpret_cast<unsigned char*>(univ->allocate(256, 4));
            std::memset(ptrs[i], i, 256);
        }
        for (int i = 0; i < 100; i++) {
            CHECK_EQ(*ptrs[i], static_cast<unsigned char>(i));
            univ->deallocate(ptrs[i], 256, 4);
        }
    }
}

TEST_SUITE_END();
