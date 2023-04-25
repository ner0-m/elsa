#pragma once
#include "ContiguousMemory.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace elsa::mr
{
    class PoolResource;

    class PoolResourceConfig
    {
    private:
        friend class PoolResource;

        size_t maxBlockSizeLog;
        size_t maxBlockSize;
        // size of the large allocation chunks that are requested from the underlying allocator
        size_t chunkSize;
        size_t maxCachedChunks;

        PoolResourceConfig(size_t maxBlockSizeLog, size_t maxBlockSize, size_t chunkSize,
                           size_t maxCachedChunks);

    public:
        /// @brief Default configuration for a pool resource with (hopefully) sensible defaults.
        /// @return Default configuration for a pool resource.
        static PoolResourceConfig defaultConfig();

        /// @brief Set the maximum size for blocks that are managed by this resource. Larger
        /// allocations are also accepted, but are serviced by the this pool's upstream allocator.
        /// @param size Maximum size for blocks managed by this pool. Must be at most as big as the
        /// chunk size. The pool resource may not use this exact value, as some minimal internal
        /// alignment requirements are applied to it.
        /// @return self
        PoolResourceConfig& setMaxBlockSize(size_t size);

        /// @brief Set the size of the chunks requested from the back-end resource. Allocations are
        /// serviced from these chunks.
        /// @param size Size of the chunks requested from the back-end resource. Must be at least
        /// as big as the maximum block size. The pool resource may not use this exact value, as
        /// some minimal internal alignment requirements are applied to it.
        /// @return self
        PoolResourceConfig& setChunkSize(size_t size);

        /// @brief  Set the maximum number of empty chunks that are cached. Further empty chunks are
        /// returned to the upstream allocator.
        /// @param count Manimum count of empty chunks to cache, before releasing memory to the
        /// upstream allocator.
        /// @return self
        PoolResourceConfig& setMaxCachedChunks(size_t count);

        /// @brief Check if the pool is misconfigured.
        /// @return true if: the configuration is valid.
        /// false if: maxBlockSize > chunkSize
        bool validate();
    };

    namespace pool_resource
    {
        // must be set to a power of 2, with sufficient unused bits for the bitfield
        const size_t BLOCK_GRANULARITY = 32;
        const size_t MIN_BLOCK_SIZE = BLOCK_GRANULARITY;
        const size_t MIN_BLOCK_SIZE_LOG = 5;
        const size_t BITFIELD_MASK = BLOCK_GRANULARITY - 1;
        const size_t SIZE_MASK = ~BITFIELD_MASK;
        const size_t FREE_BIT = 1 << 0;
        const size_t PREV_FREE_BIT = 1 << 1;

        struct Block {
            // size of the block, also storing the free and prevFree flags in its lowest two bits
            size_t _size;
            void* _address;
            // address of the block that is prior to this one in contiguous memory
            void* _prevAddress;
            // next block in the free list
            Block* _nextFree;
            // address of the previous block in the free list's next pointer
            Block** _pprevFree;

            void markFree();
            void markAllocated();

            void markPrevFree();
            void markPrevAllocated();

            bool isFree();
            bool isPrevFree();

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
        std::unordered_map<void*, std::unique_ptr<pool_resource::Block>> _addressToBlock;
        std::vector<pool_resource::Block*> _freeLists;
        uint64_t _freeListNonEmpty{0};

        size_t _cachedChunkCount;
        std::unique_ptr<std::unique_ptr<pool_resource::Block>[]> _cachedChunks;

        void insertFreeBlock(std::unique_ptr<pool_resource::Block>&& block);
        void linkFreeBlock(pool_resource::Block* block);
        void unlinkFreeBlock(pool_resource::Block* block);
        size_t freeListIndexForFreeChunk(size_t size);
        size_t computeRealSize(size_t size);
        // Returns a registered (in the address map) block that is not in the free list.
        // The metadata of the block is correctly initialized.
        pool_resource::Block* expandPool();
        void shrinkPool(std::unique_ptr<pool_resource::Block> block);
        void shrinkBlockAtTail(pool_resource::Block& block, void* blockAddress, size_t newSize,
                               size_t oldSize);

        // This function is noexcept, because it makes no allocations. The only potentially throwing
        // functions it calls, are the find and erase methods on _addressToBlock. Neither of these
        // should throw, if the inner type raises no exceptions.
        void doDeallocate(void* ptr, size_t size, size_t alignment) noexcept;

    protected:
        PoolResource(MemoryResource upstream,
                     PoolResourceConfig config = PoolResourceConfig::defaultConfig());

        ~PoolResource() = default;

    public:
        /// @brief Create a MemoryResource that encapsulates a PoolResource with the given config.
        /// @param upstream The back-end allocator that is called by the pool resource whenever it
        /// runs out of memory to service allocations.
        /// @param config The configuration for the created pool resource. It is the caller's
        /// responsibility to make sure this configuration is valid with
        /// PoolResourceConfig::validate(). If the configuration is not valid, the default
        /// configuration is used instead.
        /// @return A MemoryResource that encapsulates a PoolResource.
        static MemoryResource make(MemoryResource upstream,
                                   PoolResourceConfig config = PoolResourceConfig::defaultConfig());

        void* allocate(size_t size, size_t alignment) override;
        void deallocate(void* ptr, size_t size, size_t alignment) override;
        bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
        void copyMemory(void* ptr, const void* src, size_t size) override;
        void setMemory(void* ptr, const void* src, size_t stride, size_t count) override;
        void moveMemory(void* ptr, const void* src, size_t size) override;
    };
} // namespace elsa::mr