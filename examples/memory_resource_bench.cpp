/// Elsa example program: test execution speed of GPU projectors

#include "elsa.h"
#include "storage/memory_resource/AllocationHint.h"
#include "storage/memory_resource/ContiguousMemory.h"
#include "storage/memory_resource/UniversalResource.h"
#include "storage/memory_resource/PoolResource.h"
#include "storage/memory_resource/BitUtil.h"

#include <random>
#include <queue>
#include <chrono>
#include <iostream>
#include <vector>

using namespace elsa::mr;

struct Allocation {
    void* ptr;
    size_t size;
    size_t alignment;
    size_t priority;

    bool operator<(const Allocation& other) const
    {
        return this->priority < other.priority
               || (this->priority == other.priority
                   && reinterpret_cast<uintptr_t>(this->ptr)
                          < reinterpret_cast<uintptr_t>(other.ptr));
    }
};

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

int main()
{
    const size_t ALLOC_COUNT = 0x100000;
    std::priority_queue<Allocation> queue;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<size_t> dist;
    rng.seed(0xdeadbeef);

    MemoryResource dummy = DummyResource::make();
    MemoryResource mr = PoolResource::make(dummy);

    auto start = std::chrono::system_clock::now();
    for (size_t i = 0; i < ALLOC_COUNT; i++) {
        if (dist(rng) % 2 == 0) {
            size_t size = dist(rng) & ((static_cast<size_t>(1) << (dist(rng) % 33)) - 1);
            size_t alignment = 32;
            void* ptr = mr->allocate(size, alignment);
            queue.push(Allocation{ptr, size, alignment, dist(rng)});
        } else {
            if (!queue.empty()) {
                auto& allocation = queue.top();
                mr->deallocate(allocation.ptr, allocation.size, allocation.alignment);
                queue.pop();
            }
        }
    }
    while (!queue.empty()) {
        auto& allocation = queue.top();
        mr->deallocate(allocation.ptr, allocation.size, allocation.alignment);
        queue.pop();
    }
    auto stop = std::chrono::system_clock::now();
}
