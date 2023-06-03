#include "doctest/doctest.h"

#include "memory_resource/MemoryResource.h"
#include "memory_resource/PoolResource.h"
#include "memory_resource/UniversalResource.h"
#include "memory_resource/SyncResource.h"
#include "memory_resource/CacheResource.h"
#include "memory_resource/RegionResource.h"
#include "memory_resource/BitUtil.h"

#include <random>
#include <functional>
#include <cstring>
#include <mutex>
#include <list>
#include <set>

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

void operator delete(void* ptr, size_t size) noexcept
{
    static_cast<void>(size);
    free(ptr);
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
    static MemoryResource make()
    {
        return std::shared_ptr<MemResInterface>(new DummyResource(),
                                                [](DummyResource* p) { delete p; });
    }

    void* allocate(size_t size, size_t alignment) override
    {
        static_cast<void>(alignment);
        // this can certainly not be deref'd, as it is too large for a 48-bit sign-extended
        // address
        _allocatedSize += size;
        void* ret = _bumpPtr;
        _bumpPtr = detail::voidPtrOffset(_bumpPtr, size);
        return ret;
    }
    void deallocate(void* ptr, size_t size, size_t alignment) noexcept override
    {
        static_cast<void>(ptr);
        static_cast<void>(alignment);
        _allocatedSize -= size;
    }
    bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) noexcept override
    {
        static_cast<void>(ptr);
        static_cast<void>(size);
        static_cast<void>(alignment);
        static_cast<void>(newSize);
        return false;
    }

    size_t allocatedSize() { return _allocatedSize; }
};

static size_t sizeForRandom(size_t random)
{
    constexpr size_t MAX_POWER = 17;
    size_t power = random % MAX_POWER;
    size_t size = random & ((1 << power) - 1);
    return std::max(size, static_cast<size_t>(1));
}

static void testVaryingAllocations(MemoryResource resource)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<size_t> dist;
    rng.seed(0xdeadbeef);

    auto cmp = [](auto& a, auto& b) {
        return reinterpret_cast<uintptr_t>(a.first) < reinterpret_cast<uintptr_t>(b.first);
    };
    std::set<std::pair<void*, size_t>, decltype(cmp)> ptrs(cmp);
    for (size_t i = 0; i < 10000; i++) {
        size_t size = sizeForRandom(dist(rng));
        void* ptr = nullptr;
        try {
            ptr = resource->allocate(size, 8);
        } catch (const std::bad_alloc& e) {
            // out of memory
        }
        if (ptr) {
            auto [it, inserted] = ptrs.insert(std::make_pair(ptr, size));
            CHECK(inserted);
            if (ptrs.begin() != it) {
                auto prevIt = std::prev(it);
                /* check overlap with previous */
                REQUIRE_LE(reinterpret_cast<uintptr_t>(prevIt->first) + prevIt->second,
                           reinterpret_cast<uintptr_t>(ptr));
            }
            auto nextIt = std::next(it);
            if (nextIt != ptrs.end()) {
                /* check overlap with next */
                REQUIRE_LE(reinterpret_cast<uintptr_t>(ptr) + size,
                           reinterpret_cast<uintptr_t>(nextIt->first));
            }
        }

        constexpr size_t invDeletionProbability = 100;
        if (dist(rng) % invDeletionProbability == 0 || ptrs.size() > 3 * invDeletionProbability) {
            // E(toDelete) = invDeletionProbability => the set should not grow indefinitely
            size_t toDelete = dist(rng) % (2 * (invDeletionProbability + 1));
            for (size_t i = 0; i < toDelete && !ptrs.empty(); i++) {
                auto toDeleteIt = ptrs.begin();
                std::advance(toDeleteIt, dist(rng) % ptrs.size());
                resource->deallocate(toDeleteIt->first, toDeleteIt->second, 8);
                ptrs.erase(toDeleteIt);
            }
        }
    }

    for (auto [ptr, size] : ptrs) {
        resource->deallocate(ptr, size, 8);
    }
}

#define TEST_CASE_MR(name)                                                                        \
    TEST_CASE_TEMPLATE(name, T, ConstantFitPoolResource, FirstFitPoolResource,                    \
                       HybridFitPoolResource, UniversalResource, SyncResource<UniversalResource>, \
                       CacheResource, RegionResource)

TEST_SUITE_BEGIN("memoryresources");
TYPE_TO_STRING(ConstantFitPoolResource);
TYPE_TO_STRING(FirstFitPoolResource);
TYPE_TO_STRING(HybridFitPoolResource);
TYPE_TO_STRING(UniversalResource);
TYPE_TO_STRING(SyncResource<UniversalResource>);
TYPE_TO_STRING(CacheResource);
TYPE_TO_STRING(RegionResource);

TEST_CASE_MR("Check overlap")
{
    MemoryResource resource = T::make();
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

TEST_CASE_MR("Check alignment")
{
    MemoryResource resource = T::make();

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

TEST_CASE_MR("Varying allocations")
{
    MemoryResource baseline = globalResource();
    MemoryResource dummy = DummyResource::make();
    setGlobalResource(dummy);
    testVaryingAllocations(T::make());
    setGlobalResource(baseline);
    CHECK_EQ(dynamic_cast<DummyResource*>(dummy.get())->allocatedSize(), 0);
}

TEST_CASE_MR("Large allocation")
{
    MemoryResource resource = T::make();
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

TEST_CASE_MR("Empty allocation")
{
    MemoryResource resource = T::make();
    unsigned char* ptr = reinterpret_cast<unsigned char*>(resource->allocate(0, 32));
    // it must not be a nullptr (similar to new)
    CHECK_NE(ptr, nullptr);
    resource->deallocate(ptr, 0, 32);
}

TEST_CASE_MR("Invalid alignment")
{
    MemoryResource resource = T::make();
    CHECK_THROWS_AS(resource->allocate(32, 0), std::bad_alloc);
    CHECK_THROWS_AS(resource->allocate(32, 54), std::bad_alloc);
    CHECK_THROWS_AS(resource->allocate(32, 1023), std::bad_alloc);
}

TEST_SUITE_END();

#define TEST_CASE_POOL(name)                                                      \
    TEST_CASE_TEMPLATE(name, Pool, ConstantFitPoolResource, FirstFitPoolResource, \
                       HybridFitPoolResource)

TEST_SUITE_BEGIN("poolresource");

TEST_CASE_POOL("Allocation untouched by resource")
{
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy);
    unsigned char* ptrs[100];
    for (int i = 0; i < 100; i++) {
        ptrs[i] = reinterpret_cast<unsigned char*>(resource->allocate(256, 4));
    }
    for (int i = 0; i < 100; i++) {
        resource->deallocate(ptrs[i], 256, 4);
    }
    // nothing to check. if this test fails, it SEGFAULTs
}

TEST_CASE_POOL("Memory leak")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    // release memory to upstream immediately
    config.setMaxCachedChunks(0);

    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy, config);
    unsigned char* ptrs[109];
    for (int i = 0; i < 109; i++) {
        ptrs[i] = reinterpret_cast<unsigned char*>(resource->allocate((1 << (i % 19)) + 123, 4));
    }
    for (int i = 0, j = 0; i < 109; i++, j = (j + 311) % 109) {
        // deallocate in different order
        resource->deallocate(ptrs[j], (1 << (j % 19)) + 123, 4);
    }
    // check that there is no memory leak
    size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
    CHECK_EQ(allocatedSize, 0);
}

TEST_CASE_POOL("Resize growth")
{
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy);
    void* ptr = resource->allocate(1234, 0x10);
    CHECK(resource->tryResize(ptr, 1234, 0x10, 5678));
    testVaryingAllocations(resource);
    resource->deallocate(ptr, 1234, 0x10);
}

TEST_CASE_POOL("Resize same size")
{
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy);
    void* ptr = resource->allocate(1234, 0x10);
    CHECK(resource->tryResize(ptr, 1234, 0x10, 1234));
    // different size, but the effective size of the block is the same
    CHECK(resource->tryResize(ptr, 1234, 0x10, 1233));
    testVaryingAllocations(resource);

    resource->deallocate(ptr, 1234, 0x10);
}

TEST_CASE_POOL("Resize shrink")
{
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy);
    void* ptr = resource->allocate(1234, 0x10);
    // shrinking should always work for the pool allocator
    CHECK(resource->tryResize(ptr, 1234, 0x10, 50));
    testVaryingAllocations(resource);

    resource->deallocate(ptr, 1234, 0x10);
}

TEST_CASE_POOL("Resize failure")
{
    // This test relies on internal allocator details, so that ptr2 is always allocated directly
    // after the first
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy);
    void* ptr = resource->allocate(1234, 0x10);
    void* ptr2 = resource->allocate(1234, 0x10);
    CHECK_FALSE(resource->tryResize(ptr, 1234, 0x10, 5678));
    testVaryingAllocations(resource);
    resource->deallocate(ptr, 1234, 0x10);
    resource->deallocate(ptr2, 1234, 0x10);
}

TEST_CASE_POOL("Heap allocation fail during PoolResource::allocate")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    // release memory to upstream immediately
    config.setMaxCachedChunks(0);
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy, config);

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

    testVaryingAllocations(resource);

    for (int i = 0; i < 100; i++) {
        resource->deallocate(ptrs[i], 1234, 0x10);
    }

    size_t allocatedSize = dynamic_cast<DummyResource*>(dummy.get())->allocatedSize();
    CHECK_EQ(allocatedSize, 0);
}

TEST_CASE_POOL("PoolResource::deallocate without heap space")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    // release memory to upstream immediately
    config.setMaxCachedChunks(0);
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy, config);

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

TEST_CASE_POOL("PoolResource::tryResize without heap space")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    config.setMaxCachedChunks(0);
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy, config);

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

TEST_CASE_POOL("Cache chunk")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    config.setChunkSize(0x100);
    config.setMaxChunkSize(0x100);
    config.setMaxCachedChunks(1);
    MemoryResource dummy = DummyResource::make();
    MemoryResource resource = Pool::make(dummy, config);
    DummyResource* dummyNonOwning = dynamic_cast<DummyResource*>(dummy.get());
    GIVEN("Single chunk-sized allocation")
    {
        void* ptr = resource->allocate(0x100, 0x1);
        size_t allocatedSize = dummyNonOwning->allocatedSize();
        resource->deallocate(ptr, 0x100, 0x1);
        CHECK_EQ(allocatedSize, dummyNonOwning->allocatedSize());
    }

    GIVEN("Two chunk-sized allocations")
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

TEST_CASE_POOL("Small max chunk size")
{
    PoolResourceConfig config = PoolResourceConfig::defaultConfig();
    config.setChunkSize(0x400);
    config.setMaxChunkSize(0x1000);
    MemoryResource dummy = DummyResource::make();
    DummyResource* dummyNonOwning = dynamic_cast<DummyResource*>(dummy.get());

    GIVEN("Allocation the size of maxChunkSize")
    {
        MemoryResource resource = Pool::make(dummy, config);
        void* ptr1 = resource->allocate(0x1000, 0x1);
        CHECK_LE(dummyNonOwning->allocatedSize(), 0x1000);
        resource->deallocate(ptr1, 0x1000, 0x1);
    }

    /* Sanity check for memory leak */
    CHECK_EQ(dummyNonOwning->allocatedSize(), 0);

    GIVEN("Allocation much smaller than chunkSize")
    {
        MemoryResource resource = Pool::make(dummy, config);
        void* ptr1 = resource->allocate(0x100, 0x1);
        CHECK_EQ(dummyNonOwning->allocatedSize(), 0x400);
        resource->deallocate(ptr1, 0x100, 0x1);
    }

    /* Sanity check for memory leak */
    CHECK_EQ(dummyNonOwning->allocatedSize(), 0);

    GIVEN("Allocation larger than maxChunkSize")
    {
        MemoryResource resource = Pool::make(dummy, config);
        void* ptr1 = resource->allocate(0x10000, 0x1);
        resource->deallocate(ptr1, 0x10000, 0x1);
        /* make sure this chunk is not cached. it is too large! */
        CHECK_EQ(dummyNonOwning->allocatedSize(), 0);
    }
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("cacheresource");

TEST_CASE("an alternating allocation and deallocation pattern")
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

TEST_CASE("a limited cache resource")
{
    MemoryResource dummy = DummyResource::make();
    CacheResourceConfig config = CacheResourceConfig::defaultConfig();
    config.setMaxCacheSize(std::numeric_limits<size_t>::max());
    config.setMaxCachedCount(1);
    MemoryResource resource = CacheResource::make(dummy, config);

    std::vector<void*> ptrs(10000);

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

TEST_CASE("an unlimited cache resource")
{
    MemoryResource dummy = DummyResource::make();
    CacheResourceConfig config = CacheResourceConfig::defaultConfig();
    config.setMaxCacheSize(std::numeric_limits<size_t>::max());
    config.setMaxCachedCount(std::numeric_limits<size_t>::max());
    MemoryResource resource = CacheResource::make(dummy, config);

    std::vector<void*> ptrs(10000);

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

TEST_CASE("different allocations")
{
    MemoryResource dummy = DummyResource::make();
    CacheResourceConfig config = CacheResourceConfig::defaultConfig();
    config.setMaxCacheSize(std::numeric_limits<size_t>::max());
    config.setMaxCachedCount(1);
    MemoryResource resource = CacheResource::make(dummy, config);

    void* ptr1 = resource->allocate(0x10, 0x8);
    void* ptr2 = resource->allocate(0x100, 0x8);

    resource->deallocate(ptr1, 0x10, 0x8);
    resource->deallocate(ptr2, 0x100, 0x8);

    // check that the released chunk is the one that was deallocated first
    CHECK_EQ(0x100, dynamic_cast<DummyResource*>(dummy.get())->allocatedSize());
}

TEST_SUITE_END();
