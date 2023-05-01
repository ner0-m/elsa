#include "PoolResource.h"

#include "BitUtil.h"
#include "Assertions.h"

namespace elsa::mr
{

    PoolResourceConfig::PoolResourceConfig(size_t maxBlockSizeLog, size_t maxBlockSize,
                                           size_t chunkSize, size_t maxCachedChunks)
        : maxBlockSizeLog{maxBlockSizeLog},
          maxBlockSize{maxBlockSize},
          chunkSize{chunkSize},
          maxCachedChunks{maxCachedChunks}
    {
    }

    PoolResourceConfig PoolResourceConfig::defaultConfig()
    {
        return PoolResourceConfig(20, 1 << 20, 1 << 22, 1);
    }

    PoolResourceConfig& PoolResourceConfig::setMaxBlockSize(size_t size)
    {
        maxBlockSize = std::max(alignUp(size, pool_resource::BLOCK_GRANULARITY),
                                pool_resource::MIN_BLOCK_SIZE);
        maxBlockSizeLog = log2Floor(maxBlockSize);
        return *this;
    }

    PoolResourceConfig& PoolResourceConfig::setChunkSize(size_t size)
    {
        chunkSize = std::max(alignUp(size, pool_resource::BLOCK_GRANULARITY),
                             pool_resource::MIN_BLOCK_SIZE);
        return *this;
    }

    PoolResourceConfig& PoolResourceConfig::setMaxCachedChunks(size_t count)
    {
        maxCachedChunks = count;
        return *this;
    }

    bool PoolResourceConfig::validate()
    {
        // chunk must at least by able to accomodate the largest possible block
        return chunkSize >= maxBlockSize;
    };

    PoolResource::PoolResource(MemoryResource upstream, PoolResourceConfig config)
        : _upstream{upstream}, _freeListNonEmpty{0}, _config{config}, _cachedChunkCount{0}
    {
        if (!config.validate()) {
            _config = PoolResourceConfig::defaultConfig();
        }
        _freeLists.resize(_config.maxBlockSizeLog - pool_resource::MIN_BLOCK_SIZE_LOG, nullptr);
        _cachedChunks =
            std::make_unique<std::unique_ptr<pool_resource::Block>[]>(_config.maxCachedChunks);
    }

    MemoryResource PoolResource::make(MemoryResource upstream, PoolResourceConfig config)
    {
        return MemoryResource::MakeRef(new PoolResource(upstream, config));
    }

    void* PoolResource::allocate(size_t size, size_t alignment)
    {
        if (size == 0) {
            ++size;
        }

        if (size > _config.maxBlockSize) {
            // allocation too big to be handled by the pool
            return _upstream->allocate(size, alignment);
        }

        if (!isPowerOfTwo(alignment)) {
            throw std::bad_alloc();
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
        size_t minFreeListIndex = logBlockSize - pool_resource::MIN_BLOCK_SIZE_LOG;
        uint64_t matchingFreeListMask = ~((1 << static_cast<uint64_t>(minFreeListIndex)) - 1);
        uint64_t freeListIndex = lowestSetBit(_freeListNonEmpty & matchingFreeListMask) - 1;
        pool_resource::Block* block;
        if (freeListIndex == std::numeric_limits<uint64_t>::max()) {
            block = expandPool();
            // this block *must* fit the allocation, otherwise the requested size must be so large
            // that it is forwarded to _upstream
        } else {
            block = _freeLists[freeListIndex];
            unlinkFreeBlock(block);
        }
        // by this point, block is a registered, but unlinked block
        ENSURE(block && checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY)
               && block->size() >= realSize);

        void* retAddress = alignUp(block->_address, alignment);
        size_t headSplitSize =
            reinterpret_cast<uintptr_t>(retAddress) - reinterpret_cast<uintptr_t>(block->_address);
        size_t remainingSize = block->size() - headSplitSize;

        if (headSplitSize != 0) {
            // insert new block after head
            pool_resource::Block* newBlock;
            try {
                auto newBlockOwned = std::make_unique<pool_resource::Block>();
                newBlock = newBlockOwned.get();
                _addressToBlock.insert({retAddress, std::move(newBlockOwned)});
            } catch (...) {
                // make sure the failed allocation has no side effects (aside from potentially
                // enlarging the pool)
                linkFreeBlock(block);
                throw std::bad_alloc();
            }

            block->setSize(headSplitSize);
            // block is already in the map, but must be reinserted (into an appropriate free list)
            linkFreeBlock(block);

            // size and free bit are set below
            newBlock->markPrevFree();
            newBlock->_address = retAddress;
            newBlock->_prevAddress = block->_address;
            block = newBlock;
        }

        block->markAllocated();
        block->setSize(remainingSize);
        try {
            shrinkBlockAtTail(*block, retAddress, realSize, remainingSize);
            return retAddress;
        } catch (...) {
            // this is safe to call, because it is noexcept and should free block, re-merging it
            // with its predecessor
            doDeallocate(block->_address, size, alignment);
            throw std::bad_alloc();
        }
    }

    void PoolResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        if (!ptr) {
            return;
        }
        if (size > _config.maxBlockSize) {
            // allocation too big to be handled by the pool, must have come from upstream
            _upstream->deallocate(ptr, size, alignment);
            return;
        }
        doDeallocate(ptr, size, alignment);
    }

    void PoolResource::doDeallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        auto blockIt = _addressToBlock.find(ptr);
        ENSURE(blockIt != _addressToBlock.end());
        pool_resource::Block* block = blockIt->second.get();
        block->markFree();

        auto prevIt = _addressToBlock.find(block->_prevAddress);
        if (prevIt != _addressToBlock.end() && prevIt->second->isFree()) {
            // coalesce with prev. block
            pool_resource::Block* prev = prevIt->second.get();
            unlinkFreeBlock(prev);
            prev->setSize(prev->size() + block->size());
            _addressToBlock.erase(blockIt);
            block = prev;
        }

        void* nextAdress = voidPtrOffset(block->_address, block->size());
        auto nextIt = _addressToBlock.find(nextAdress);
        if (nextIt != _addressToBlock.end() && nextIt->second->isFree()
            && nextIt->second->_prevAddress
                   != nullptr) { // _prevAddress == nullptr indicates that the block is the
                                 // start block of another chunk, which just happens to be next
                                 // to this one. Never coalesce accross chunk boundaries
            // coalesce with next block
            pool_resource::Block* next = nextIt->second.get();
            unlinkFreeBlock(next);
            block->setSize(block->size() + next->size());
            _addressToBlock.erase(nextIt);
        }

        nextAdress = voidPtrOffset(block->_address, block->size());
        nextIt = _addressToBlock.find(nextAdress);
        if (nextIt != _addressToBlock.end() && nextIt->second->_prevAddress != nullptr) {
            std::unique_ptr<pool_resource::Block>& next = nextIt->second;
            next->markPrevFree();
            next->_prevAddress = block->_address;
        }
        if (block->size() < _config.chunkSize) {
            linkFreeBlock(block);
        } else {
            auto blockIt = _addressToBlock.find(block->_address);
            shrinkPool(std::move(blockIt->second));
            _addressToBlock.erase(blockIt);
        }
    }

    bool PoolResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        if (!ptr || size > _config.maxBlockSize) {
            return false;
        }
        auto blockIt = _addressToBlock.find(ptr);
        ENSURE(blockIt != _addressToBlock.end());
        std::unique_ptr<pool_resource::Block>& block = blockIt->second;

        size_t realSize = computeRealSize(newSize);
        size_t currentSize = block->size();
        if (realSize == currentSize) {
            return true;
        } else if (realSize > currentSize) {
            void* nextAdress = voidPtrOffset(ptr, currentSize);
            auto nextIt = _addressToBlock.find(nextAdress);
            if (nextIt != _addressToBlock.end() && nextIt->second->isFree()
                && nextIt->second->_prevAddress != nullptr) {
                std::unique_ptr<pool_resource::Block>& next = nextIt->second;
                size_t cumulativeSize = currentSize + next->size();
                if (cumulativeSize >= realSize) {
                    unlinkFreeBlock(next.get());
                    std::unique_ptr<pool_resource::Block> next = std::move(nextIt->second);
                    try {
                        _addressToBlock.erase(nextIt);
                    } catch (...) {
                        // Erase should never be able to throw here, so if this is reached we are in
                        // dire straits
                        ENSURE(0, "Unreachable!");
                    }
                    block->setSize(realSize);
                    if (cumulativeSize > realSize) {
                        next->_address = voidPtrOffset(ptr, realSize);
                        next->setSize(cumulativeSize - realSize);
                        try {
                            insertFreeBlock(std::move(next));
                        } catch (...) {
                            // In case we cannot insert the new free block into the map/free-list,
                            // have block subsume it. This may cause some very undesirable internal
                            // fragmentation, but it is better than leaking this memory.
                            block->setSize(cumulativeSize);
                        }
                    }
                    return true;
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            try {
                shrinkBlockAtTail(*block, ptr, realSize, currentSize);
            } catch (...) {
                return false;
            }
            return true;
        }
    }

    void PoolResource::copyMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->copyMemory(ptr, src, size);
    }

    void PoolResource::setMemory(void* ptr, const void* src, size_t stride, size_t count)
    {
        _upstream->setMemory(ptr, src, stride, count);
    }

    void PoolResource::moveMemory(void* ptr, const void* src, size_t size)
    {
        _upstream->moveMemory(ptr, src, size);
    }

    void PoolResource::insertFreeBlock(std::unique_ptr<pool_resource::Block>&& block)
    {
        // if insert throws, this is side-effect free
        auto [it, _] = _addressToBlock.insert({block->_address, std::move(block)});
        linkFreeBlock(it->second.get());
    }

    void PoolResource::linkFreeBlock(pool_resource::Block* block)
    {
        size_t size = block->size();
        ENSURE(checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY)
               && size % pool_resource::BLOCK_GRANULARITY == 0);
        size_t freeListIndex = freeListIndexForFreeChunk(block->size());
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
        // as a result of this, allocations are sometimes served from a larger free list, even
        // if a smaller one would fit. However, this categorization saves us from having to
        // iterate through free lists, looking for the best fit
        uint64_t freeListLog = log2Floor(size);
        size_t freeListIndex = freeListLog - pool_resource::MIN_BLOCK_SIZE_LOG;
        // all blocks larger than the size for the largest free list are stored there as well
        freeListIndex = std::min(freeListIndex, _freeLists.size() - 1);
        return freeListIndex;
    }

    size_t PoolResource::computeRealSize(size_t size)
    {
        return (size + pool_resource::BLOCK_GRANULARITY - 1)
               & ~(pool_resource::BLOCK_GRANULARITY - 1);
    }

    pool_resource::Block* PoolResource::expandPool()
    {
        void* newChunkAddress;
        std::unique_ptr<pool_resource::Block> newChunk;
        if (_cachedChunkCount == 0) {
            newChunkAddress =
                _upstream->allocate(_config.chunkSize, pool_resource::BLOCK_GRANULARITY);
            newChunk = std::make_unique<pool_resource::Block>();
            newChunk->_address = newChunkAddress;
            newChunk->_size = _config.chunkSize | pool_resource::FREE_BIT;
            newChunk->_prevAddress = nullptr;
        } else {
            newChunk = std::move(_cachedChunks[--_cachedChunkCount]);
            newChunkAddress = newChunk->_address;
        }

        try {
            auto [it, _] = _addressToBlock.insert({newChunkAddress, std::move(newChunk)});
            return it->second.get();
        } catch (...) {
            _upstream->deallocate(newChunkAddress, _config.chunkSize,
                                  pool_resource::BLOCK_GRANULARITY);
            throw;
        }
    }

    void PoolResource::shrinkPool(std::unique_ptr<pool_resource::Block> chunk)
    {
        if (_cachedChunkCount < _config.maxCachedChunks) {
            _cachedChunks[_cachedChunkCount++] = std::move(chunk);
        } else {
            _upstream->deallocate(chunk->_address, _config.chunkSize,
                                  pool_resource::BLOCK_GRANULARITY);
        }
    }

    void PoolResource::shrinkBlockAtTail(pool_resource::Block& block, void* blockAddress,
                                         size_t newSize, size_t oldSize)
    {
        // oldSize and newSize must be multiples of pool_resource::BLOCK_GRANULARITY
        size_t tailSplitSize = oldSize - newSize;
        if (tailSplitSize != 0) {
            // insert sub-block after the allocation into a free-list
            try {
                auto successor = std::make_unique<pool_resource::Block>();
                successor->_size = tailSplitSize | pool_resource::FREE_BIT;
                successor->_address = voidPtrOffset(blockAddress, newSize);
                successor->_prevAddress = blockAddress;
                insertFreeBlock(std::move(successor));
            } catch (...) {
                throw;
            }

            block.setSize(newSize);
        }
    }

    namespace pool_resource
    {
        void Block::markFree()
        {
            _size |= FREE_BIT;
        }

        void Block::markAllocated()
        {
            _size &= ~FREE_BIT;
        }

        void Block::markPrevFree()
        {
            _size |= PREV_FREE_BIT;
        }

        void Block::markPrevAllocated()
        {
            _size &= ~PREV_FREE_BIT;
        }

        bool Block::isFree()
        {
            return _size & FREE_BIT;
        }

        bool Block::isPrevFree()
        {
            return _size & PREV_FREE_BIT;
        }

        void Block::unlinkFree()
        {
            *_pprevFree = _nextFree;
            if (_nextFree) {
                _nextFree->_pprevFree = _pprevFree;
            }
        }

        void Block::insertAfterFree(Block** pprev)
        {
            _pprevFree = pprev;
            _nextFree = *pprev;
            *pprev = this;
            if (_nextFree) {
                _nextFree->_pprevFree = &_nextFree;
            }
        }

        size_t Block::size()
        {
            return _size & SIZE_MASK;
        }

        void Block::setSize(size_t size)
        {
            ENSURE((size & BITFIELD_MASK) == 0);
            _size = (_size & BITFIELD_MASK) | size;
        }

    } // namespace pool_resource
} // namespace elsa::mr
