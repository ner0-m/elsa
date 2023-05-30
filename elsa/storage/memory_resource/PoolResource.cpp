#include "PoolResource.h"

#include "BitUtil.h"
#include "Assertions.h"

/*
 * Free list layout:
 * let k = log2(MIN_BLOCK_SIZE)
 *
 * | n | -> [2^{k+n}..]
 * | . |
 * | 2 | -> [2^{k+2}..2^{k+3} - 1]
 * | 1 | -> [2^{k+1}..2^{k+2} - 1]
 * | 0 | -> [2^k    ..2^{k+1} - 1]
 *
 * Constant time allocation strategy:
 * - allocate from _freeLists[ceil(log2(size))]
 * - guaranteed fit
 * - downside: expect more fragmentation, e.g. when freeing and
 *   allocating a block of equal size and alignment, the block
 *   that was just freed is potentially not considered.
 *   Larger blocks may become progressively more fragmented over
 *   time.
 *
 * First fit allocation strategy:
 * - allocate from _freeLists[floor(log2(size))]
 * - linearly search for the first sufficiently large block in
 *   the list
 *
 * For all strategies:
 * - insert into _freeLists[floor(log2(size))]
 * - allocate from a larger free list, if the originally chosen
 *   one is empty/cannot service the allocation
 */

namespace elsa::mr
{

    PoolResourceConfig::PoolResourceConfig(size_t maxChunkSize, size_t chunkSize,
                                           size_t maxCachedChunks)
        : maxChunkSize{maxChunkSize}, chunkSize{chunkSize}, maxCachedChunks{maxCachedChunks}
    {
    }

    PoolResourceConfig PoolResourceConfig::defaultConfig()
    {
        return PoolResourceConfig(static_cast<size_t>(1) << 33, static_cast<size_t>(1) << 22, 1);
    }

    PoolResourceConfig& PoolResourceConfig::setMaxChunkSize(size_t size)
    {
        maxChunkSize = std::max(alignUp(size, pool_resource::BLOCK_GRANULARITY),
                                pool_resource::MIN_BLOCK_SIZE);
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
        return chunkSize <= maxChunkSize && chunkSize >= pool_resource::MIN_BLOCK_SIZE;
    };

    namespace pool_resource
    {
        template <typename FreeListStrategy>
        PoolResource<FreeListStrategy>::PoolResource(MemoryResource upstream,
                                                     PoolResourceConfig config)
            : _upstream{upstream}, _config{config}
        {
            if (!config.validate()) {
                _config = PoolResourceConfig::defaultConfig();
            }
            _freeLists.resize(
                log2Floor(_config.maxChunkSize) - pool_resource::MIN_BLOCK_SIZE_LOG + 1, nullptr);
            _cachedChunks =
                std::make_unique<std::unique_ptr<pool_resource::Block>[]>(_config.maxCachedChunks);
        }

        template <typename FreeListStrategy>
        PoolResource<FreeListStrategy>::~PoolResource()
        {
            for (size_t i = 0; i < _cachedChunkCount; i++) {
                _upstream->deallocate(_cachedChunks[i]->_address, _cachedChunks[i]->_chunkSize,
                                      pool_resource::BLOCK_GRANULARITY);
            }
        }

        template <typename FreeListStrategy>
        MemoryResource PoolResource<FreeListStrategy>::make(MemoryResource upstream,
                                                            PoolResourceConfig config)
        {
            return MemoryResource::MakeRef(new PoolResource(upstream, config));
        }

        template <typename FreeListStrategy>
        void* PoolResource<FreeListStrategy>::allocate(size_t size, size_t alignment)
        {
            auto [realSize, blockSize] = computeSizeWithAlginment(size, alignment);
            if (blockSize > _config.maxChunkSize) {
                // allocation too big to be handled by the pool
                return _upstream->allocate(size, alignment);
            }

            pool_resource::Block* block =
                FreeListStrategy::selectBlock(_freeListNonEmpty, _freeLists, blockSize);
            if (!block) {
                block = expandPool(blockSize);
                // this block *must* fit the allocation, otherwise the requested size must be so
                // large that it is forwarded to _upstream
            } else {
                unlinkFreeBlock(block);
            }
            // by this point, block is a registered, but unlinked block
            ENSURE(block && checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY)
                   && block->size() >= realSize);

            void* retAddress = alignUp(block->_address, alignment);
            size_t headSplitSize = reinterpret_cast<uintptr_t>(retAddress)
                                   - reinterpret_cast<uintptr_t>(block->_address);
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
                // block is already in the map, but must be reinserted (into an appropriate free
                // list)
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
                doDeallocate(block->_address);
                throw std::bad_alloc();
            }
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::deallocate(void* ptr, size_t size,
                                                        size_t alignment) noexcept
        {
            if (!ptr) {
                return;
            }
            // This throws, if alignment is not a power of two. Due to the noexcept tag,
            // this leads to termination.
            // This behavior seems appropriate, since this is essentially an invalid free.
            // The provided pointer cannot possible be allocated with that alignment.
            // If this behavior is undesired, let me know.
            auto [_, blockSize] = computeSizeWithAlginment(size, alignment);
            if (blockSize > _config.maxChunkSize) {
                // allocation too big to be handled by the pool, must have come from upstream
                _upstream->deallocate(ptr, size, alignment);
                return;
            }
            doDeallocate(ptr);
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::doDeallocate(void* ptr) noexcept
        {
            auto blockIt = _addressToBlock.find(ptr);
            ENSURE(blockIt != _addressToBlock.end());
            pool_resource::Block* block = blockIt->second.get();
            block->markFree();

            if (!block->isChunkStart()) {
                auto prevIt = _addressToBlock.find(block->_prevAddress);
                if (prevIt != _addressToBlock.end() && prevIt->second->isFree()) {
                    // coalesce with prev. block
                    pool_resource::Block* prev = prevIt->second.get();
                    unlinkFreeBlock(prev);
                    prev->setSize(prev->size() + block->size());
                    _addressToBlock.erase(blockIt);
                    block = prev;
                }
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
            if (nextIt != _addressToBlock.end() && !nextIt->second->isChunkStart()) {
                std::unique_ptr<pool_resource::Block>& next = nextIt->second;
                next->markPrevFree();
                next->_prevAddress = block->_address;
            }
            if (block->isChunkStart() && block->size() == block->_chunkSize) {
                auto blockIt = _addressToBlock.find(block->_address);
                shrinkPool(std::move(blockIt->second));
                _addressToBlock.erase(blockIt);
            } else {
                linkFreeBlock(block);
            }
        }

        template <typename FreeListStrategy>
        bool PoolResource<FreeListStrategy>::tryResize(void* ptr, size_t size, size_t alignment,
                                                       size_t newSize)
        {
            static_cast<void>(size);
            static_cast<void>(alignment);
            if (!ptr || size > _config.maxChunkSize) {
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
                            // Erase should never be able to throw here, so if this is reached we
                            // are in dire straits
                            ENSURE(0, "Unreachable!");
                        }
                        block->setSize(realSize);
                        if (cumulativeSize > realSize) {
                            next->_address = voidPtrOffset(ptr, realSize);
                            next->setSize(cumulativeSize - realSize);
                            try {
                                insertFreeBlock(std::move(next));
                            } catch (...) {
                                // In case we cannot insert the new free block into the
                                // map/free-list, have block subsume it. This may cause some very
                                // undesirable internal fragmentation, but it is better than leaking
                                // this memory.
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

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::insertFreeBlock(
            std::unique_ptr<pool_resource::Block>&& block)
        {
            // if insert throws, this is side-effect free
            auto [it, _] = _addressToBlock.insert({block->_address, std::move(block)});
            linkFreeBlock(it->second.get());
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::linkFreeBlock(pool_resource::Block* block)
        {
            size_t size = block->size();
            ENSURE(checkAlignment(block->_address, pool_resource::BLOCK_GRANULARITY)
                   && size % pool_resource::BLOCK_GRANULARITY == 0);
            uint64_t freeListLog = log2Floor(size);
            size_t freeListIndex = freeListLog - pool_resource::MIN_BLOCK_SIZE_LOG;
            // all blocks larger than the size for the largest free list are stored there as well
            freeListIndex = std::min(freeListIndex, _freeLists.size() - 1);
            block->insertAfterFree(&_freeLists[freeListIndex]);
            _freeListNonEmpty |= 1 << freeListIndex;
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::unlinkFreeBlock(pool_resource::Block* block)
        {
            block->unlinkFree();
            size_t freeListIndex = freeListIndexForFreeChunk(block->size());
            // if the free list is now empty, mark it in the bitmask
            if (_freeLists[freeListIndex] == nullptr) {
                _freeListNonEmpty &= ~(1 << freeListIndex);
            }
        }

        template <typename FreeListStrategy>
        size_t PoolResource<FreeListStrategy>::freeListIndexForFreeChunk(size_t size)
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

        template <typename FreeListStrategy>
        size_t PoolResource<FreeListStrategy>::computeRealSize(size_t size)
        {
            return (size + pool_resource::BLOCK_GRANULARITY - 1)
                   & ~(pool_resource::BLOCK_GRANULARITY - 1);
        }

        template <typename FreeListStrategy>
        std::pair<size_t, size_t>
            PoolResource<FreeListStrategy>::computeSizeWithAlginment(size_t requestedSize,
                                                                     size_t requestedAlignment)
        {
            if (requestedSize == 0) {
                ++requestedSize;
            }

            if (!isPowerOfTwo(requestedAlignment)) {
                throw std::bad_alloc();
            }

            // find best-fitting non-empty bin
            size_t realSize = computeRealSize(requestedSize);

            // this overflow check is probably unnecessary, since the log is already compared
            // against the max block size
            ENSURE(realSize >= requestedSize);
            // minimal size of the free block to carve the allocation out of. must be enough for to
            // contain an aligned allocation
            size_t blockSize;
            if (requestedAlignment <= pool_resource::BLOCK_GRANULARITY) {
                blockSize = realSize;
            } else {
                blockSize = realSize + requestedAlignment;
                ENSURE(blockSize >= realSize);
            }
            return {realSize, blockSize};
        }

        template <typename FreeListStrategy>
        pool_resource::Block* PoolResource<FreeListStrategy>::expandPool(size_t requestedSize)
        {
            void* newChunkAddress;
            std::unique_ptr<pool_resource::Block> newChunk = nullptr;
            for (size_t i = 0; i < _cachedChunkCount; i++) {
                if (_cachedChunks[i]->size() >= requestedSize) {
                    --_cachedChunkCount;
                    std::swap(_cachedChunks[i], _cachedChunks[_cachedChunkCount]);
                    newChunk = std::move(_cachedChunks[_cachedChunkCount]);
                    newChunkAddress = newChunk->_address;
                    break;
                }
            }

            if (!newChunk) {
                // This should be enforced in allocate, by forwarding to _upstream
                ENSURE(requestedSize <= _config.maxChunkSize);
                // Rationale for the chunk size: if a chunk of this size is requested,
                // another chunk of similar size will likely be requested soon. With this
                // choice of chunkSize, 4 such allocations can be served without allocating
                // from _upstream again
                size_t chunkSize =
                    std::min(std::max(requestedSize << 2, _config.chunkSize), _config.maxChunkSize);
                chunkSize = alignUp(chunkSize, pool_resource::BLOCK_GRANULARITY);
                // May throw std::bad_alloc
                newChunkAddress = _upstream->allocate(chunkSize, pool_resource::BLOCK_GRANULARITY);
                newChunk = std::make_unique<pool_resource::Block>();
                newChunk->_address = newChunkAddress;
                newChunk->_size =
                    chunkSize | pool_resource::FREE_BIT | pool_resource::CHUNK_START_BIT;
                newChunk->_chunkSize = chunkSize;
            }

            try {
                auto [it, _] = _addressToBlock.insert({newChunkAddress, std::move(newChunk)});
                return it->second.get();
            } catch (...) {
                _upstream->deallocate(newChunkAddress, newChunk->_chunkSize,
                                      pool_resource::BLOCK_GRANULARITY);
                throw;
            }
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::shrinkPool(std::unique_ptr<pool_resource::Block> chunk)
        {
            if (_cachedChunkCount < _config.maxCachedChunks) {
                _cachedChunks[_cachedChunkCount++] = std::move(chunk);
            } else {
                _upstream->deallocate(chunk->_address, chunk->_chunkSize,
                                      pool_resource::BLOCK_GRANULARITY);
            }
        }

        template <typename FreeListStrategy>
        void PoolResource<FreeListStrategy>::shrinkBlockAtTail(pool_resource::Block& block,
                                                               void* blockAddress, size_t newSize,
                                                               size_t oldSize)
        {
            // oldSize and newSize must be multiples of pool_resource::BLOCK_GRANULARITY
            size_t tailSplitSize = oldSize - newSize;
            if (tailSplitSize != 0) {
                void* subblockAddress = voidPtrOffset(blockAddress, newSize);
                // insert sub-block after the allocation into a free-list
                try {
                    auto subblock = std::make_unique<pool_resource::Block>();
                    subblock->_size = tailSplitSize | pool_resource::FREE_BIT;
                    subblock->_address = subblockAddress;
                    subblock->_prevAddress = blockAddress;
                    insertFreeBlock(std::move(subblock));
                } catch (...) {
                    throw;
                }

                block.setSize(newSize);
                // Current state:
                //      .---_prevAddress----.
                //      v                   |
                // +---------+---------+-----------+
                // | A       | A'      | B         |
                // +---------+---------+-----------+
                // Where A is the original block, A' is the subblock created in
                // its tail, and B is the successor.
                // So, the successor's (B) _prevAddress must be adjusted
                // (if there is one and we are not at the end of the chunk)
                void* successorAddress = voidPtrOffset(blockAddress, oldSize);
                auto successorIt = _addressToBlock.find(successorAddress);
                if (successorIt != _addressToBlock.end()) {
                    successorIt->second->_prevAddress = subblockAddress;
                }
            }
        }

        void Block::markFree()
        {
            _size |= FREE_BIT;
        }

        void Block::markAllocated()
        {
            _size &= ~FREE_BIT;
        }

        void Block::markChunkStart()
        {
            _size &= ~CHUNK_START_BIT;
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

        bool Block::isChunkStart()
        {
            return _size & CHUNK_START_BIT;
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

        pool_resource::Block*
            ConstantTimeFit::selectBlock(uint64_t listOccupancy,
                                         const std::vector<pool_resource::Block*>& freeLists,
                                         size_t blockSize)
        {
            size_t logBlockSize = log2Ceil(blockSize);
            size_t minFreeListIndex = logBlockSize - pool_resource::MIN_BLOCK_SIZE_LOG;
            uint64_t matchingFreeListMask = ~((1 << static_cast<uint64_t>(minFreeListIndex)) - 1);
            uint64_t freeListIndex = lowestSetBit(listOccupancy & matchingFreeListMask) - 1;
            if (freeListIndex == std::numeric_limits<uint64_t>::max()) {
                return nullptr;
            } else {
                return freeLists[freeListIndex];
            }
        }

        pool_resource::Block*
            FirstFit::selectBlock(uint64_t listOccupancy,
                                  const std::vector<pool_resource::Block*>& freeLists,
                                  size_t blockSize)
        {
            size_t logBlockSize = log2Floor(blockSize);
            size_t minFreeListIndex = logBlockSize - pool_resource::MIN_BLOCK_SIZE_LOG;
            uint64_t matchingFreeListMask = ~((1 << static_cast<uint64_t>(minFreeListIndex)) - 1);
            uint64_t freeListIndex = lowestSetBit(listOccupancy & matchingFreeListMask) - 1;
            if (freeListIndex == std::numeric_limits<uint64_t>::max()) {
                return nullptr;
            } else if (freeListIndex == minFreeListIndex) {
                // first fit search
                for (pool_resource::Block* block = freeLists[freeListIndex]; block;
                     block = block->_nextFree) {
                    if (block->size() >= blockSize) {
                        return block;
                    }
                }
                matchingFreeListMask &= ~(1 << static_cast<uint64_t>(freeListIndex));
                freeListIndex = lowestSetBit(listOccupancy & matchingFreeListMask);
                if (freeListIndex == std::numeric_limits<uint64_t>::max()) {
                    return freeLists[freeListIndex];
                } else {
                    return nullptr;
                }
            } else {
                return freeLists[freeListIndex];
            }
        }

        pool_resource::Block*
            HybridFit::selectBlock(uint64_t listOccupancy,
                                   const std::vector<pool_resource::Block*>& freeLists,
                                   size_t blockSize)
        {
            pool_resource::Block* block =
                ConstantTimeFit::selectBlock(listOccupancy, freeLists, blockSize);
            if (block) {
                return block;
            }

            size_t logBlockSize = log2Floor(blockSize);
            size_t freeListIndex = logBlockSize - pool_resource::MIN_BLOCK_SIZE_LOG;
            for (block = freeLists[freeListIndex]; block; block = block->_nextFree) {
                if (block->size() >= blockSize) {
                    return block;
                }
            }

            return nullptr;
        }

        template class PoolResource<ConstantTimeFit>;
        template class PoolResource<FirstFit>;
        template class PoolResource<HybridFit>;
    } // namespace pool_resource
} // namespace elsa::mr
