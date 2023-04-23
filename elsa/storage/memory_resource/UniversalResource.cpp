#include "UniversalResource.h"
#include <memory>

// TODO: find a way to determine the presence of cuda
#if 0
#include <cuda_runtime.h>
namespace elsa::mr
{
    namespace universal_resource
    {
        // minimal alignment of pointers returned by any of the cuda malloc calls
        // according to the cuda C programming guide, section 5.3.2 version 12.1
        constexpr size_t GUARANTEED_ALIGNMENT = 256;

        struct ChunkHeader {
            uintptr_t offset;
        }
    } // namespace universal_resource

    void* UniversalResource::allocate(size_t size, size_t alignment)
    {
        if (!alignment || (alignment & (alignment - 1))) [[unlikely]] {
            // alignment is not a power of 2
            throw std::bad_alloc();
        }
        if (alignment > universal_resource::GUARANTEED_ALIGNMENT) {
            size_t sizeWithAlignment = size + alignment;
            size_t totalSize = sizeWithAlignment + sizeof(universal_resource::ChunkHeader);
            if (sizeWithAlignment < size || totalSize < sizeWithAlignment) [[unlikely]] {
                throw std::bad_alloc();
            }
            uintptr_t ptr;
            if (cudaMallocManaged(reinterpret_cast<void**>(&ptr), size)) [[unlikely]] {
                throw std::bad_alloc();
            }
            uintptr_t retPtr = (ptr + totalSize - 1) & alignment - 1;
            universal_resource::ChunkHeader* hdr =
                static_cast<universal_resource::ChunkHeader*>(retPtr - sizeof(chunkHdr));
            hdr->offset = retPtr - ptr;

            return static_cast<void*>(retPtr);
        } else {
            void* ptr;
            if (cudaMallocManaged(&ptr, size)) [[unlikely]] {
                throw std::bad_alloc();
            }
            return ptr;
        }
    }

    void* UniversalResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        static_cast<void>(size);
        if (alignment > universal_resource::GUARANTEED_ALIGNMENT) {
            uintptr_t ptr = static_cast<uintptr_t>(ptr);
            universal_resource::ChunkHeader* hdr = static_cast<universal_resource::ChunkHeader*>(
                ptr - sizeof(universal_resource::ChunkHeader));
            void* allocatedPtr = static_cast<void*>(ptr - hdr->offset);
            cudaFree(allocatedPtr);
        } else {
            cudaFree(ptr);
        }
    }

    bool UniversalResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                                      size_t newAlignment)
    {
        return false;
    }

    void UniversalResource::copyMemory(void* ptr, const void* src, size_t size) noexcept
    {
        //TODO
    }

    void UniversalResource::setMemory(void* ptr, const void* src, size_t stride,
                                      size_t count) noexcept
    {
        // TODO
    }
    void UniversalResource::moveMemory(void* ptr, const void* src, size_t size) noexcept
    {
        // TODO
    }
} // namespace elsa::mr
#else
#include <stdlib.h>
#include <cstring>

namespace elsa::mr
{
    // no cuda => regular heap allocation

    void* elsa::mr::UniversalResource::allocate(size_t size, size_t alignment)
    {
        void* ptr = std::aligned_alloc(size, alignment);
        if (!ptr) [[unlikely]] {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void UniversalResource::deallocate(void* ptr, size_t size, size_t alignment)
    {
        static_cast<void>(size);
        static_cast<void>(alignment);
        std::free(ptr);
    }

    bool UniversalResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize,
                                      size_t newAlignment)
    {
        return false;
    }

    void UniversalResource::copyMemory(void* ptr, const void* src, size_t size) noexcept
    {
        std::memcpy(ptr, src, size);
    }

    void UniversalResource::setMemory(void* ptr, const void* src, size_t stride,
                                      size_t count) noexcept
    {
        // TODO: optimize this
        for (size_t i = 0; i < count; i++) {
            std::memcpy(ptr, src, stride);
            ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + stride);
        }
    }
    void UniversalResource::moveMemory(void* ptr, const void* src, size_t size) noexcept
    {
        std::memmove(ptr, src, size);
    }
} // namespace elsa::mr
#endif
