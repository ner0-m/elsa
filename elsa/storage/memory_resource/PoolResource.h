#pragma once
#include "ContiguousMemory.h"
#include <unordered_map>
#include <vector>
#include <memory>

namespace elsa::mr
{
    namespace pool_resource
    {
        template <typename T>
        class PoolResource;
    }

    class PoolResourceConfig
    {
    private:
        template <typename T>
        friend class pool_resource::PoolResource;

        size_t maxChunkSize;
        // size of the large allocation chunks that are requested from the underlying allocator
        size_t chunkSize;
        size_t maxCachedChunks;

        PoolResourceConfig(size_t maxChunkSize, size_t chunkSize, size_t maxCachedChunks);

    public:
        /// @brief Default configuration for a pool resource with (hopefully) sensible defaults.
        /// @return Default configuration for a pool resource.
        static PoolResourceConfig defaultConfig();

        /// @brief Set the maximum size for chunks allocated by this resource.
        /// Chunks are regions of memory, from which the blocks that are returned by allocate()
        /// are suballocated. Hence, this value also limits the size of blocks that are managed
        /// by this resource. Larger allocations are also accepted, but are serviced by the this
        /// pool's upstream allocator.
        /// @param size Maximum size for blocks managed by this pool.
        /// The pool resource may not use this exact value, as some minimal internal
        /// alignment requirements are applied to it.
        /// @return self
        PoolResourceConfig& setMaxChunkSize(size_t size);

        /// @brief Set the size of the chunks requested from the back-end resource. Allocations are
        /// serviced from these chunks.
        /// @param size Size of the chunks requested from the back-end resource. Must be at most
        /// as big as the maximum chunk size. The pool resource may not use this exact value, as
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
        /// false if: chunkSize > maxChunkSize
        bool validate();
    };

    namespace pool_resource
    {
        // must be set to a power of 2, with sufficient unused bits for the bitfield
        // => granularity must be at least 8!
        // 256 is chosen to make sure types for cuda kernels are sufficiently aligned.
        const size_t BLOCK_GRANULARITY = 256;
        const size_t MIN_BLOCK_SIZE = BLOCK_GRANULARITY;
        const size_t MIN_BLOCK_SIZE_LOG = 8;
        const size_t BITFIELD_MASK = BLOCK_GRANULARITY - 1;
        const size_t SIZE_MASK = ~BITFIELD_MASK;
        const size_t FREE_BIT = 1 << 0;
        const size_t PREV_FREE_BIT = 1 << 1;
        const size_t CHUNK_START_BIT = 1 << 2;

        struct Block {
            // size of the block, also storing the free and prevFree flags in its lowest two bits
            size_t _size;
            void* _address;
            union {
                // if this block marks the start of a chunk, this field contains its size
                size_t _chunkSize;
                // if this block is not the beginning of a chunk, this the contains address
                // of the block that is prior to this one in contiguous memory
                void* _prevAddress;
            };
            // next block in the free list
            Block* _nextFree;
            // address of the previous block in the free list's next pointer
            Block** _pprevFree;

            void markFree();
            void markAllocated();
            void markChunkStart();

            void markPrevFree();
            void markPrevAllocated();

            bool isFree();
            bool isPrevFree();
            bool isChunkStart();

            void unlinkFree();
            void insertAfterFree(Block** pprev);

            size_t size();
            void setSize(size_t size);
        };

        template <typename FreeListStrategy>
        class PoolResource : public MemResInterface
        {
        private:
            MemoryResource _upstream;
            PoolResourceConfig _config;
            std::unordered_map<void*, std::unique_ptr<pool_resource::Block>> _addressToBlock;
            std::vector<pool_resource::Block*> _freeLists;
            uint64_t _freeListNonEmpty{0};

            size_t _cachedChunkCount{0};
            std::unique_ptr<std::unique_ptr<pool_resource::Block>[]> _cachedChunks;

            void insertFreeBlock(std::unique_ptr<pool_resource::Block>&& block);
            void linkFreeBlock(pool_resource::Block* block);
            void unlinkFreeBlock(pool_resource::Block* block);
            size_t freeListIndexForFreeChunk(size_t size);
            size_t computeRealSize(size_t size);
            // pair element 1: realSize, i.e. the size of the block to return
            // pair element 2: blockSize, i.e. the required size for a block,
            //                 so that the allocation can be carved out of it.
            //                 blockSize >= realSize holds, to satisfy alignment guarantees.
            std::pair<size_t, size_t> computeSizeWithAlginment(size_t requestedSize,
                                                               size_t requestedAlignment);
            // Returns a registered (in the address map) block that is not in the free list.
            // The metadata of the block is correctly initialized.
            pool_resource::Block* expandPool(size_t requestedSize);
            void shrinkPool(std::unique_ptr<pool_resource::Block> block);
            void shrinkBlockAtTail(pool_resource::Block& block, void* blockAddress, size_t newSize,
                                   size_t oldSize);

            // This function is noexcept, because it makes no allocations. The only potentially
            // throwing functions it calls, are the find and erase methods on _addressToBlock.
            // Neither of these should throw, if the inner type raises no exceptions.
            void doDeallocate(void* ptr) noexcept;

            PoolResource(const PoolResource& other) = delete;
            PoolResource& operator=(const PoolResource& other) = delete;
            PoolResource(PoolResource&& other) noexcept = delete;
            PoolResource& operator=(PoolResource&& other) noexcept = delete;

        protected:
            PoolResource(MemoryResource upstream,
                         PoolResourceConfig config = PoolResourceConfig::defaultConfig());

            ~PoolResource();

        public:
            /// @brief Create a MemoryResource that encapsulates a PoolResource with the given
            /// config.
            /// @param upstream The back-end allocator that is called by the pool resource whenever
            /// it runs out of memory to service allocations.
            /// @param config The configuration for the created pool resource. It is the caller's
            /// responsibility to make sure this configuration is valid with
            /// PoolResourceConfig::validate(). If the configuration is not valid, the default
            /// configuration is used instead.
            /// @return A MemoryResource that encapsulates a PoolResource.
            static MemoryResource
                make(MemoryResource upstream,
                     PoolResourceConfig config = PoolResourceConfig::defaultConfig());

            void* allocate(size_t size, size_t alignment) override;
            void deallocate(void* ptr, size_t size, size_t alignment) noexcept override;
            bool tryResize(void* ptr, size_t size, size_t alignment, size_t newSize) override;
        };

        struct ConstantTimeFit {
            static pool_resource::Block*
                selectBlock(uint64_t listOccupancy,
                            const std::vector<pool_resource::Block*>& freeLists, size_t blockSize);
        };

        struct FirstFit {
            static pool_resource::Block*
                selectBlock(uint64_t listOccupancy,
                            const std::vector<pool_resource::Block*>& freeLists, size_t blockSize);
        };

        struct HybridFit {
            static pool_resource::Block*
                selectBlock(uint64_t listOccupancy,
                            const std::vector<pool_resource::Block*>& freeLists, size_t blockSize);
        };
    } // namespace pool_resource

    using PoolResource = pool_resource::PoolResource<pool_resource::HybridFit>;

    /// @brief Pool resource able to serve allocations in average constant time (provided the
    /// upstream allocator also gives this guarantee). May lead to more fragmentation than a first
    /// fit strategy.
    using ConstantFitPoolResource = pool_resource::PoolResource<pool_resource::ConstantTimeFit>;

    /// @brief Pool resource that serves allocations via searching the corresponding seg. free list
    /// in linear time.
    using FirstFitPoolResource = pool_resource::PoolResource<pool_resource::FirstFit>;

    /// @brief Pool resource follows the constant time fit strategy, but falls back to linear time
    /// search when no block can be found in the larger lists.
    using HybridFitPoolResource = pool_resource::PoolResource<pool_resource::HybridFit>;
} // namespace elsa::mr
