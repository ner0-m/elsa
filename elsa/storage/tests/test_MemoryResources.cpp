#include "doctest/doctest.h"

#include "memory_resource/ContiguousMemory.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/UniversalResource.h"
#include "memory_resource/SynchResource.h"
#include "memory_resource/CacheResource.h"
#include "memory_resource/BitUtil.h"

#include "Assertions.h"

#include <random>
#include <functional>
#include <cstring>
#include <mutex>
#include <list>

using namespace elsa::mr;

bool failNextAlloc = false;

void* operator new(size_t size)
{
    if (failNextAlloc) {
        failNextAlloc = false;
        throw std::bad_alloc{};
    }

    if (size == 0)
        ++size; // don't malloc(0)

    if (void* ptr = std::malloc(size))
        return ptr;

    throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept
{
    free(ptr);
}

void operator delete(void* ptr, size_t size)
{
    static_cast<void>(size);
    free(ptr);
}

template <typename T>
static MemoryResource provideResource()
{
    ENSURE(false);
    return defaultInstance();
}

template <>
MemoryResource provideResource<PoolResource>()
{
    MemoryResource upstream = UniversalResource::make();
    return PoolResource::make(upstream);
}

template <>
MemoryResource provideResource<UniversalResource>()
{
    return UniversalResource::make();
}

template <>
MemoryResource provideResource<SynchResource<PoolResource>>()
{
    MemoryResource upstream = UniversalResource::make();
    return SynchResource<PoolResource>::make(upstream);
}

template <>
MemoryResource provideResource<CacheResource>()
{
    MemoryResource upstream = UniversalResource::make();
    return PoolResource::make(upstream);
}

static size_t sizeForIndex(size_t i)
{
    constexpr size_t MAX_POWER = 20;
    size_t hash = std::hash<size_t>{}(i);
    size_t power = hash % MAX_POWER;
    size_t size = hash & ((1 << power) - 1);
    return std::max(size, static_cast<size_t>(1));
}

// call this to check whether the basic allocator functionality still works,
// i.e. that the metadata is still intact
// (e.g. after peforming some special operation)
static void testNonoverlappingAllocations(MemoryResource resource)
{
    const size_t NUM_ALLOCATIONS = 128;
    std::vector<std::pair<void*, size_t>> allocations;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<size_t> dist;
    rng.seed(0xdeadbeef);

    for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
        // first pick the bin at random
        size_t size = (2 << (dist(rng) % 17));
        // then pick a random size within that bin
        size |= (size - 1) & dist(rng);
        void* ptr = resource->allocate(size, 4);
        allocations.emplace_back(ptr, size);
    }

    std::sort(allocations.begin(), allocations.end(), [](auto& a, auto& b) {
        return reinterpret_cast<uintptr_t>(a.first) < reinterpret_cast<uintptr_t>(b.first);
    });

    // check if any allocations overlap
    for (size_t i = 1; i < NUM_ALLOCATIONS; i++) {
        auto [ptr1, size] = allocations[i - 1];
        auto [ptr2, _] = allocations[i];
        CHECK_LE(reinterpret_cast<uintptr_t>(ptr1) + size, reinterpret_cast<uintptr_t>(ptr2));
    }

    for (size_t i = 0; i < NUM_ALLOCATIONS; i++) {
        auto [ptr, size] = allocations[i];
        resource->deallocate(ptr, size, 4);
    }
}

TEST_SUITE_BEGIN("memoryresources");

TEST_CASE_TEMPLATE("Memory resource", T, PoolResource, UniversalResource,
                   SynchResource<PoolResource>, CacheResource)
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
            CHECK_EQ(*(ptrs[i] + 255), static_cast<unsigned char>(i));
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

class DummyResource : public MemResInterface
{
private:
    void* _bumpPtr;
    size_t _allocatedSize{0};

protected:
    DummyResource() : _bumpPtr{reinterpret_cast<void*>(0xfffffffffff000)} {}
    ~DummyResource() = default;

public:
    static MemoryResource make() { return MemoryResource::MakeRef(new DummyResource()); }

    void* allocate(size_t size, size_t alignment) override
    {
        static_cast<void>(alignment);
        // this can certainly not be deref'd, as it is too large for a 48-bit sign-extended
        // address
        _allocatedSize += size;
        void* ret = _bumpPtr;
        _bumpPtr = voidPtrOffset(_bumpPtr, size);
        return ret;
    }
    void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
    {
        static_cast<void>(ptr);
        static_cast<void>(alignment);
        _allocatedSize -= size;
    }
    bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override
    {
        static_cast<void>(ptr);
        static_cast<void>(size);
        static_cast<void>(alignment);
        static_cast<void>(newSize);
        return false;
    }

    size_t allocatedSize() { return _allocatedSize; }
};

TEST_CASE("Pool resource")
{

    GIVEN("Allocation untouched by resource")
    {
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy);
        unsigned char* ptrs[100];
        for (int i = 0; i < 100; i++) {
            ptrs[i] = reinterpret_cast<unsigned char*>(resource->allocate(256, 4));
        }
        for (int i = 0; i < 100; i++) {
            resource->deallocate(ptrs[i], 256, 4);
        }
        // nothing to check. if this test fails, it SEGFAULTs
    }

    GIVEN("Memory leak")
    {
        PoolResourceConfig config = PoolResourceConfig::defaultConfig();
        // release memory to upstream immediately
        config.setMaxCachedChunks(0);

        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy, config);
        unsigned char* ptrs[109];
        for (int i = 0; i < 109; i++) {
            ptrs[i] =
                reinterpret_cast<unsigned char*>(resource->allocate((1 << (i % 19)) + 123, 4));
        }
        for (int i = 0, j = 0; i < 109; i++, j = (j + 311) % 109) {
            // deallocate in different order
            resource->deallocate(ptrs[j], (1 << (j % 19)) + 123, 4);
        }
        // check that there is no memory leak
        size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
        CHECK_EQ(allocatedSize, 0);
    }

    GIVEN("Resize growth")
    {
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy);
        void* ptr = resource->allocate(1234, 0x10);
        CHECK(resource->tryResize(ptr, 1234, 0x10, 5678));
        testNonoverlappingAllocations(resource);
        resource->deallocate(ptr, 1234, 0x10);
    }

    GIVEN("Resize same size")
    {
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy);
        void* ptr = resource->allocate(1234, 0x10);
        CHECK(resource->tryResize(ptr, 1234, 0x10, 1234));
        // different size, but the effective size of the block is the same
        CHECK(resource->tryResize(ptr, 1234, 0x10, 1233));
        testNonoverlappingAllocations(resource);

        resource->deallocate(ptr, 1234, 0x10);
    }

    GIVEN("Resize shrink")
    {
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy);
        void* ptr = resource->allocate(1234, 0x10);
        // shrinking should always work for the pool allocator
        CHECK(resource->tryResize(ptr, 1234, 0x10, 50));
        testNonoverlappingAllocations(resource);

        resource->deallocate(ptr, 1234, 0x10);
    }

    GIVEN("Resize failure")
    {
        // This test relies on internal allocator details, so that ptr2 is always allocated directly
        // after the first
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy);
        void* ptr = resource->allocate(1234, 0x10);
        void* ptr2 = resource->allocate(1234, 0x10);
        CHECK_FALSE(resource->tryResize(ptr, 1234, 0x10, 5678));
        testNonoverlappingAllocations(resource);
        resource->deallocate(ptr, 1234, 0x10);
        resource->deallocate(ptr2, 1234, 0x10);
    }

    GIVEN("Heap allocation fail during PoolResource::allocate")
    {
        PoolResourceConfig config = PoolResourceConfig::defaultConfig();
        // release memory to upstream immediately
        config.setMaxCachedChunks(0);
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy, config);

        void* ptrs[100];
        for (int i = 0; i < 50; i++) {
            ptrs[i] = resource->allocate(1234, 0x10);
        }

        failNextAlloc = true;

        int exceptions = 0;
        for (int i = 50; i < 100; i++) {
            try {
                ptrs[i] = resource->allocate(1234, 0x10);
            } catch (std::bad_alloc& e) {
                ++exceptions;
                ptrs[i] = nullptr;
            }
        }
        CHECK_EQ(exceptions, 1);

        testNonoverlappingAllocations(resource);

        for (int i = 0; i < 100; i++) {
            resource->deallocate(ptrs[i], 1234, 0x10);
        }

        size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
        CHECK_EQ(allocatedSize, 0);
    }

    GIVEN("PoolResource::deallocate without heap space")
    {
        PoolResourceConfig config = PoolResourceConfig::defaultConfig();
        // release memory to upstream immediately
        config.setMaxCachedChunks(0);
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy, config);

        void* ptrs[100];
        for (int i = 0; i < 100; i++) {
            ptrs[i] = resource->allocate(1234, 0x10);
        }

        failNextAlloc = true;

        for (int i = 0; i < 100; i++) {
            resource->deallocate(ptrs[i], 1234, 0x10);
        }

        int* ptr = nullptr;
        CHECK_THROWS(ptr = new int);
        delete ptr;

        // check that there is no memory leak
        size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
        CHECK_EQ(allocatedSize, 0);
    }

    GIVEN("PoolResource::tryResize without heap space")
    {
        PoolResourceConfig config = PoolResourceConfig::defaultConfig();
        config.setMaxCachedChunks(0);
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy, config);

        void* ptr = resource->allocate(0x1000, 0x10);

        failNextAlloc = true;
        CHECK_NOTHROW(resource->tryResize(ptr, 0x1000, 0x10, 0x10));
        failNextAlloc = true;
        CHECK_NOTHROW(resource->tryResize(ptr, 0x1000, 0x10, 0x10000));
        failNextAlloc = false;
        resource->deallocate(ptr, 0x1000, 0x10);

        size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
        CHECK_EQ(allocatedSize, 0);
    }

    GIVEN("Cache chunk")
    {
        PoolResourceConfig config = PoolResourceConfig::defaultConfig();
        config.setChunkSize(0x100);
        config.setMaxBlockSize(0x100);
        config.setMaxCachedChunks(1);
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = PoolResource::make(dummy, config);
        DummyResource* dummyNonOwning = dynamic_cast<DummyResource*>(dummy.get());
        {
            void* ptr = resource->allocate(0x100, 0x1);
            size_t allocatedSize = dummyNonOwning->allocatedSize();
            resource->deallocate(ptr, 0x100, 0x1);
            CHECK_EQ(allocatedSize, dummyNonOwning->allocatedSize());
        }
        {
            void* ptr1 = resource->allocate(0x100, 0x1);
            size_t sizeWithOneChunk = dummyNonOwning->allocatedSize();
            void* ptr2 = resource->allocate(0x100, 0x1);
            size_t sizeWithTwoChunks = dummyNonOwning->allocatedSize();
            resource->deallocate(ptr1, 0x100, 0x1);
            resource->deallocate(ptr2, 0x100, 0x1);
            CHECK_LT(sizeWithOneChunk, sizeWithTwoChunks);
            CHECK_EQ(sizeWithOneChunk, dummyNonOwning->allocatedSize());
        }
    }
}

TEST_CASE("Cache resource")
{
    GIVEN("an alternating allocation and deallocation pattern")
    {
        MemoryResource dummy = DummyResource::make();
        MemoryResource resource = CacheResource::make(dummy);
        for (int i = 0; i < 10; i++) {
            size_t size = 0x100 + 0x10 * i;
            size_t alignment = 1 << i;
            resource->deallocate(resource->allocate(size, alignment), size, alignment);
        }
        size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
        void* ptrs[10];
        for (int i = 0; i < 10; i++) {
            size_t size = 0x100 + 0x10 * i;
            size_t alignment = 1 << i;
            ptrs[i] = resource->allocate(size, alignment);
        }
        // no additional allocations
        CHECK_EQ(allocatedSize, dynamic_cast<DummyResource*>(dummy.get())->allocatedSize());
        for (int i = 0; i < 10; i++) {
            size_t size = 0x100 + 0x10 * i;
            size_t alignment = 1 << i;
            resource->deallocate(ptrs[i], size, alignment);
        }
    }

    GIVEN("a limited cache resource") {
        MemoryResource dummy = DummyResource::make();
        CacheResourceConfig config = CacheResourceConfig::defaultConfig();
        config.setMaxCacheSize(std::numeric_limits<size_t>::max());
        config.setMaxCachedCount(1);
        MemoryResource resource = CacheResource::make(dummy, config);

        void* ptrs[10000];

        size_t allocationSize = 0x1000000;
        for (int i = 0; i < 10000; i++) {
            ptrs[i] = resource->allocate(allocationSize, 0x8);
        }
        for (int i = 0; i < 10000; i++) {
            resource->deallocate(ptrs[i], allocationSize, 0x8);
        }

        // all but one entries released
        CHECK_EQ(allocationSize, dynamic_cast<DummyResource*>(dummy.get())->allocatedSize());
    }

    GIVEN("an unlimited cache resource") {
        MemoryResource dummy = DummyResource::make();
        CacheResourceConfig config = CacheResourceConfig::defaultConfig();
        config.setMaxCacheSize(std::numeric_limits<size_t>::max());
        config.setMaxCachedCount(std::numeric_limits<size_t>::max());
        MemoryResource resource = CacheResource::make(dummy, config);

        void* ptrs[10000];

        size_t allocationSize = 0x1000000;
        for (int i = 0; i < 10000; i++) {
            ptrs[i] = resource->allocate(allocationSize, 0x8);
        }
        for (int i = 0; i < 10000; i++) {
            resource->deallocate(ptrs[i], allocationSize, 0x8);
        }

        // no memory released, all cached
        CHECK_EQ(10000 * allocationSize, dynamic_cast<DummyResource*>(dummy.get())->allocatedSize());
    }
}

TEST_SUITE_END();
