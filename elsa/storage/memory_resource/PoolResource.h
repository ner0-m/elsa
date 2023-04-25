#pragma once
#include "ContiguousMemory.h"
#include <unordered_map>
#include <vector>

namespace elsa::mr
{
    struct PoolResourceConfig {
        size_t maxBlockSizeLog;
        size_t maxBlockSize;

        static PoolResourceConfig defaultConfig()
        {
            return PoolResourceConfig{.maxBlockSizeLog = 20, .maxBlockSize = 1 << 20};
        }
    };

    namespace pool_resource
    {
        // must be set to a power of 2, with sufficient unused bits for the bitfield
        const size_t BLOCK_GRANULARITY = 32;
        const size_t MIN_BLOCK_SIZE = BLOCK_GRANULARITY;
        const size_t MIN_BLOCK_SIZE_LOG = 5;
        const size_t BITFIELD_MASK = BLOCK_GRANULARITY - 1;
        const size_t SIZE_MASK = ~BITFIELD_MASK;

        struct Block {
            union {
                size_t _size;
                struct {
                    size_t _isFree : 1;
                    size_t _isPrevFree : 1;
                };
            };
            void* _address;
            // address of the block that is prior to this one in contiguous memory
            void* _prevAddress;
            // next block in the free list
            Block* _nextFree;
            // address of the previous block in the free list's next pointer
            Block** _pprevFree;

            bool isFree();

            void unlinkFree();
            void insertAfterFree(Block** pprev);

            size_t size();
            void setSize(size_t size);
        };
    } // namespace pool_resource

    class PoolResource : public MemResInterface
    {
    private:
        MemoryResource _upstream;
        PoolResourceConfig _config;
        std::unordered_map<void*, pool_resource::Block*> _addressToBlock;
        std::vector<pool_resource::Block*> _freeLists;
        uint64_t _freeListNonEmpty;

        void insertFreeBlock(pool_resource::Block* block);
        void unlinkFreeBlock(pool_resource::Block* block);
        size_t freeListIndexForFreeChunk(size_t size);
        size_t computeRealSize(size_t size);
        void expandPool();

    public:
        PoolResource(MemoryResource upstream,
                     PoolResourceConfig config = PoolResourceConfig::defaultConfig());

        ~PoolResource() = default;

        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                       size_t newAlignment) override;

        void copyMemory(void* ptr, const void* src, size_t size) override;
        void setMemory(void* ptr, const void* src, size_t stride, size_t count) override;
        void moveMemory(void* ptr, const void* src, size_t size) override;
    };
} // namespace elsa::mr