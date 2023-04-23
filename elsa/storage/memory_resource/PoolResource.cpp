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

        void* retAddress = alignUp(block->_address, alignment);
        size_t headSplitSize =
            reinterpret_cast<uintptr_t>(retAddress) - reinterpret_cast<uintptr_t>(block->_address);
        size_t remainingSize = block->size() - headSplitSize;
        if (headSplitSize != 0) {
            block->setSize(headSplitSize);
            // insert new block after head
            pool_resource::Block* newBlock = new pool_resource::Block();
            // size and free bit are set below
            newBlock->_isPrevFree = 1;
            newBlock->_address = retAddress;
            newBlock->_prevAddress = block->_address;
            auto [newBlockIt, _] = _addressToBlock.insert({retAddress, newBlock});
            block = newBlock;
        } else {
            unlinkFreeBlock(block);
        }

        block->_isFree = 0;
        block->setSize(realSize);

        // remainingSize must be a multiple of pool_resource::BLOCK_GRANULARITY
        size_t tailSplitSize = remainingSize - realSize;
        if (tailSplitSize != 0) {
            // insert sub-block after the allocation into a free-list
            pool_resource::Block* successor = new pool_resource::Block();
            successor->setSize(tailSplitSize);
            successor->_address =
                reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(retAddress) + realSize);
            successor->_isPrevFree = 0;
            successor->_prevAddress = retAddress;
            insertFreeBlock(successor);
        }
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
        auto blockIt = _addressToBlock.find(ptr);
        ENSURE(blockIt != _addressToBlock.end());
        pool_resource::Block* block = blockIt->second;

        auto prevIt = _addressToBlock.find(block->_prevAddress);
        if (prevIt != _addressToBlock.end() && prevIt->second->_isFree) {
            // coalesce with prev. block
            pool_resource::Block* prev = prevIt->second;
            unlinkFreeBlock(prev);
            prev->setSize(prev->size() + block->size());
            _addressToBlock.erase(blockIt);
            delete block;
            block = prev;
        }

        void* nextAdress =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(block->_address) + block->size());
        auto nextIt = _addressToBlock.find(nextAdress);
        if (nextIt != _addressToBlock.end() && nextIt->second->_isFree) {
            // coalesce with next block
            pool_resource::Block* next = nextIt->second;
            unlinkFreeBlock(next);
            block->setSize(block->size() + next->size());
            _addressToBlock.erase(nextIt);
            delete next;
        }

        void* nextAdress =
            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(block->_address) + block->size());
        auto nextIt = _addressToBlock.find(nextAdress);
        if (nextIt != _addressToBlock.end()) {
            pool_resource::Block* next = nextIt->second;
            next->_isPrevFree = 1;
            next->_prevAddress = block->_address;
        }

        insertFreeBlock(block);
    }

    void PoolResource::insertFreeBlock(pool_resource::Block* block)
    {
        size_t size = block->size();
        ENSURE(checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY)
               && size % pool_resource::BLOCK_GRANULARITY == 0);
        size_t freeListIndex = freeListIndexForFreeChunk(block->size());
        auto [it, _] = _addressToBlock.insert({block->_address, block});
        block->insertAfterFree(&_freeLists[freeListIndex]);
        _freeListNonEmpty |= 1 << freeListIndex;
    }

    void PoolResource::unlinkFreeBlock(pool_resource::Block* block)
    {
        block->unlinkFree();
        size_t freeListIndex = freeListIndexForFreeChunk(block->size());
        // if the free list is now empty, mark it in the bitmask
        if (_freeLists[freeListIndex] == nullptr) {
            _freeListNonEmpty &= ~(1 << freeListIndex);
        }
    }

    size_t PoolResource::freeListIndexForFreeChunk(size_t size)
    {
        // the largest power of 2 that fits into size determines the free list.
        // as a result of this, allocations are sometimes served from a larger free list, even if a
        // smaller one would fit. However, this categorization saves us from having to iterate
        // through free lists, looking for the best fit
        uint64_t freeListLog = log2Floor(size);
        size_t freeListIndex = freeListLog - _config.minBlockSizeLog;
        // all blocks larger than the size for the largest free list are stored there as well
        freeListIndex = std::min(freeListIndex, _freeLists.size() - 1);
        return freeListIndex;
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
