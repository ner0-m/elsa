#include "PoolResource.h"
#include <bit>
#include "Assertions.h"

namespace elsa::mr
{
    template <typename T>
    static T log2Floor(T t)
    {
        return std::bit_width(t) - 1;
    }

    // does not work for t == 0
    template <typename T>
    static T log2Ceil(T t)
    {
        return std::bit_width(t - 1);
    }

    template <typename T>
    static T lowestSetBit(T t)
    {
        return std::bit_width(t & ~(t - 1));
    }

    // alignment must be a power of 2
    static bool checkAlignment(void* ptr, size_t alignment)
    {
        return ptr == alignDown(ptr, alignment);
    }

    // alignment must be a power of 2
    static void* alignDown(void* ptr, size_t alignment)
    {
        return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) & ~(alignment - 1));
    }

    // alignment must be a power of 2
    static void* alignUp(void* ptr, size_t alignment)
    {
        return alignDown(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + alignment - 1),
                         alignment);
    }

    PoolResource::~PoolResource()
    {
        _upstream->releaseRef();
    }

    PoolResource::PoolResource(MemoryResource* upstream, PoolResourceConfig config)
        : _upstream{upstream}, _config{config}
    {
        upstream->addRef();
    }

    void* PoolResource::allocate(size_t size, size_t alignment)
    {
        if (size == 0)
            return nullptr;
        size_t logSize = log2Ceil(size);
        if (logSize > _config.maxBlockSizeLog) {
            // allocation too big to be handled by the pool
            return _upstream->allocate(size, alignment);
        }

        // find best-fitting non-empty bin
        size_t realSize = computeRealSize(size);

        // this overflow check is probably unnecessary, since the log is already compared against
        // the max block size
        ENSURE(realSize >= size);
        // minimal size of the free block to carve the allocation out of. must be enough for to
        // contain an aligned allocation
        size_t blockSize;
        if (alignment <= pool_resource::BLOCK_GRANULARITY) {
            blockSize = realSize;
        } else {
            blockSize = realSize + alignment;
            ENSURE(blockSize >= realSize);
        }

        size_t logBlockSize = log2Ceil(blockSize);
        size_t minFreeListIndex = logBlockSize - _config.minBlockSizeLog;
        uint64_t matchingFreeListMask = ~((1 << static_cast<uint64_t>(minFreeListIndex)) - 1);
        uint64_t freeListIndex = lowestSetBit(_freeListNonEmpty & matchingFreeListMask) - 1;
        if (freeListIndex == std::numeric_limits<uint64_t>::max()) [[unlikely]] {
            // TODO: allocate new pool space
            throw std::bad_alloc();
        }
        pool_resource::Block* block = _freeLists[freeListIndex];
        ENSURE(block && checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY));
        block->unlinkFree();
        // if the free list is now empty, mark it in the bitmask
        if (_freeLists[freeListIndex] == nullptr) {
            _freeListNonEmpty &= ~(1 << freeListIndex);
        }

        void* retAddress = alignUp(block->_address, alignment);
        size_t newPredecessorSize =
            reinterpret_cast<uintptr_t>(retAddress) - reinterpret_cast<uintptr_t>(block->_address);
        if (newPredecessorSize != 0) {
            // insert sub-block preceeding the allocation into a free-list
            pool_resource::Block predecessor;
            predecessor.setSize(newPredecessorSize);
            predecessor._address = block->_address;
            predecessor._prevAddress = block->_prevAddress;
            insertFreeBlock(predecessor);
        }
        size_t remainingSize = block->size() - newPredecessorSize;
        // remainingSize must be a multiple of pool_resource::BLOCK_GRANULARITY
        size_t tailSize = remainingSize - realSize;
        if (tailSize != 0) {
            // insert sub-block after the allocation into a free-list
            pool_resource::Block successor;
            successor.setSize(tailSize);
            successor._address =
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(retAddress) + realSize);
            successor._prevAddress = retAddress;
            insertFreeBlock(successor);
        }
        ENSURE(_addressToBlock.erase(block->_address) == 1);
        return retAddress;
    }

    void PoolResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        size_t logSize = log2Ceil(size);
        if (logSize > _config.maxBlockSizeLog) {
            // allocation too big to be handled by the pool, must have come from upstream
            _upstream->deallocate(ptr, size, alignment);
            return;
        }
        size_t realSize = computeRealSize(size);
        pool_resource::Block block;
        block._address = ptr;
        block._prevAddress = nullptr;
        block._size = realSize;
        auto prevIt = _addressToBlock.find(block._prevAddress);

        pool_resource::Block* blockPtr;
        if (prevIt != _addressToBlock.end()) {

        } else {
            auto [blockIt, _] = _addressToBlock.insert({block._address, block});
            blockPtr = &blockIt->second;
        }
    }

    void PoolResource::insertFreeBlock(pool_resource::Block block)
    {
        size_t size = block.size();
        ENSURE(checkAlignment(block._address, pool_resource::BLOCK_GRANULARITY)
               && size % pool_resource::BLOCK_GRANULARITY == 0);
        // the largest power of 2 that fits into size determines the free list.
        // as a result of this, allocations are sometimes served from a larger free list, even if a
        // smaller one would fit. However, this categorization saves us from having to iterate
        // through free lists, looking for the best fit
        size_t freeListLog = log2Floor(size);
        size_t freeListIndex = freeListLog - _config.minBlockSizeLog;
        auto [it, _] = _addressToBlock.insert({block._address, block});
        pool_resource::Block* inserted = &it->second;
        inserted->insertAfterFree(&_freeLists[freeListIndex]);
        _freeListNonEmpty |= 1 << freeListIndex;
    }

    size_t PoolResource::computeRealSize(size_t size)
    {
        return (size + pool_resource::BLOCK_GRANULARITY - 1) & ~pool_resource::BLOCK_GRANULARITY;
    }
} // namespace elsa::mr

bool elsa::mr::pool_resource::Block::isFree()
{
    return isFree;
}

void elsa::mr::pool_resource::Block::unlinkFree()
{
    *_pprevFree = _nextFree;
    if (_nextFree) {
        _nextFree->_pprevFree = _pprevFree;
    }
}

void elsa::mr::pool_resource::Block::insertAfterFree(Block** pprev)
{
    _pprevFree = pprev;
    _nextFree = *pprev;
    *pprev = this;
    if (_nextFree) {
        _nextFree->_pprevFree = &_nextFree;
    }
}

size_t elsa::mr::pool_resource::Block::size()
{
    return _size & SIZE_MASK;
}

void elsa::mr::pool_resource::Block::setSize(size_t size)
{
    ENSURE((size & BITFIELD_MASK) == 0);
    _size = _size & BITFIELD_MASK | size;
}
