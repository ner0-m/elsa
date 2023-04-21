#include "UniversalResource.h"
#include <memory>

// TODO: find better way to determine the presence of cuda
#if ELSA_CUDA_PROJECTORS
#include <cuda_runtime.h>
namespace elsa::mr
{
    namespace universal_allocator
    {
        struct ChunkHeader {
            uintptr_t offset;
        }
    } // namespace universal_allocator

    // TODO: optimize out header data structure when alignment is already implicitly fulfilled (e.g.
    // alignment == 1)
    void* UniversalResource::allocate(size_t size, size_t alignment)
    {
        if (!alignment || (alignment & (alignment - 1))) [[unlikely]] {
            // alignment is not a power of 2
            throw std::bad_alloc();
        }
        size_t sizeWithAlignment = size + alignment;
        size_t totalSize = sizeWithAlignment + sizeof(universal_allocator::ChunkHeader);
        if (sizeWithAlignment < size || totalSize < sizeWithAlignment) [[unlikely]] {
            throw std::bad_alloc();
        }
        uintptr_t ptr;
        if (cudaMallocManaged(reinterpret_cast<void**>(&ptr), size)) [[unlikely]] {
            throw std::bad_alloc();
        }
        uintptr_t retPtr = (ptr + totalSize - 1) & alignment - 1;
        universal_allocator::ChunkHeader* hdr =
            static_cast<universal_allocator::ChunkHeader*>(retPtr - sizeof(chunkHdr));
        hdr->offset = retPtr - ptr;

        return static_cast<void*>(retPtr);
    }

    void* UniversalResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        uintptr_t ptr = static_cast<uintptr_t>(ptr);
        universal_allocator::ChunkHeader* hdr = static_cast<universal_allocator::ChunkHeader*>(
            ptr - sizeof(universal_allocator::ChunkHeader));
        void* allocatedPtr = static_cast<void*>(ptr - hdr->offset);
        cudaFree(allocatedPtr);
    }
} // namespace elsa::mr
#else
#include <stdlib.h>

namespace elsa::mr
{
    // no cuda => regular heap allocation

    void* elsa::mr::UniversalResource::allocate(size_t size, size_t alignment)
    {
        void* ptr;
        if (!posix_memalign(&ptr, size, alignment)) [[unlikely]]
            throw std::bad_alloc();
        return ptr;
        // return operator new(size, alignment);
    }

    void UniversalResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        // TODO: freeing memory allocated with posix_memalign is not portable
        free(ptr);
        // operator delete(ptr);
    }
} // namespace elsa::mr
#endif
