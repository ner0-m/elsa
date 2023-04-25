#include "doctest/doctest.h"

#include "memory_resource/ContiguousMemory.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/UniversalResource.h"

#include "Assertions.h"

#include <random>
#include <functional>
#include <cstring>

using namespace elsa::mr;

template <typename T>
static MemoryResource provideResource()
{
    ENSURE(false);
    return defaultInstance();
}

template <>
MemoryResource provideResource<PoolResource>()
{
    HostStandardResource* upstream = new HostStandardResource();
    MemoryResource upstreamRef = MemoryResource::MakeRef(upstream);
    return MemoryResource::MakeRef(new PoolResource(upstreamRef));
}

template <>
MemoryResource provideResource<UniversalResource>()
{
    return MemoryResource::MakeRef(new UniversalResource());
}

static size_t sizeForIndex(size_t i)
{
    constexpr size_t MAX_POWER = 20;
    size_t hash = std::hash<size_t>{}(i);
    size_t power = hash % MAX_POWER;
    size_t size = hash & ((1 << power) - 1);
    return std::max(size, static_cast<size_t>(1));
}

TEST_SUITE_BEGIN("memoryresources");

TEST_CASE_TEMPLATE("Memory resource", T, PoolResource, UniversalResource)
{
    GIVEN("Check overlap")
    {
        MemoryResource resource = provideResource<T>();
        unsigned char* ptrs[100];
        for (int i = 0; i < 100; i++) {
            ptrs[i] = reinterpret_cast<unsigned char*>(resource->allocate(256, 4));
            std::memset(ptrs[i], i, 256);
        }
        for (int i = 0; i < 100; i++) {
            CHECK_EQ(*ptrs[i], static_cast<unsigned char>(i));
            resource->deallocate(ptrs[i], 256, 4);
        }
    }

    GIVEN("Check alignment")
    {
        MemoryResource resource = provideResource<T>();

        void* ptrs[12];
        for (size_t i = 0, alignment = 1; i < 12; i++, alignment <<= 1) {
            ptrs[i] = resource->allocate(256, alignment);
            uintptr_t ptr = reinterpret_cast<uintptr_t>(ptrs[i]);
            CHECK_EQ(ptr & (alignment - 1), 0);
        }
        for (size_t i = 0, alignment = 1; i < 12; i++, alignment <<= 1) {
            resource->deallocate(ptrs[i], 256, alignment);
        }
    }

    GIVEN("Varying allocations")
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<size_t> dist;
        rng.seed(0xdeadbeef);

        MemoryResource resource = provideResource<T>();
        unsigned char* ptrs[100];
        for (size_t i = 0; i < 100; i++) {
            size_t size = sizeForIndex(i);
            ptrs[i] = reinterpret_cast<unsigned char*>(resource->allocate(size, 8));
            *ptrs[i] = 0xff;
            if (i > 0 && dist(rng) % 2 == 0) {
                size_t indexToDeallocate = dist(rng) % i;
                size_t size = sizeForIndex(indexToDeallocate);
                resource->deallocate(ptrs[indexToDeallocate], size, 8);
                ptrs[indexToDeallocate] = nullptr;
            }
        }

        for (size_t i = 0; i < 100; i++) {
            size_t size = sizeForIndex(i);
            resource->deallocate(ptrs[i], size, 8);
        }
    }

    GIVEN("Large allocation")
    {
        MemoryResource resource = provideResource<T>();
        // allocation may fail by std::bad_alloc, just as long as it does not fail in some
        // unspecified way
        try {
            unsigned char* ptr = reinterpret_cast<unsigned char*>(resource->allocate(1 << 30, 32));
            // if a pointer is returned, it must be valid
            CHECK_NE(ptr, nullptr);
            ptr[0] = 'A';
            ptr[1 << 29] = 'B';
            resource->deallocate(ptr, 1 << 30, 32);
        } catch (std::bad_alloc& e) {
        }
    }

    GIVEN("Empty allocation")
    {
        MemoryResource resource = provideResource<T>();
        unsigned char* ptr = reinterpret_cast<unsigned char*>(resource->allocate(0, 32));
        // it must not be a nullptr (similar to new)
        CHECK_NE(ptr, nullptr);
        resource->deallocate(ptr, 0, 32);
    }

    GIVEN("Empty allocation")
    {
        MemoryResource resource = provideResource<T>();
        unsigned char* ptr = reinterpret_cast<unsigned char*>(resource->allocate(0, 32));
        // it must not be a nullptr (similar to new)
        CHECK_NE(ptr, nullptr);
        resource->deallocate(ptr, 0, 32);
    }

    GIVEN("Invalid alignment")
    {
        MemoryResource resource = provideResource<T>();
        CHECK_THROWS_AS(resource->allocate(32, 0), std::bad_alloc);
        CHECK_THROWS_AS(resource->allocate(32, 54), std::bad_alloc);
        CHECK_THROWS_AS(resource->allocate(32, 1023), std::bad_alloc);
    }
}

TEST_SUITE_END();
