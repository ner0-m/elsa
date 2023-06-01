#ifdef ELSA_CUDA_ENABLED
#include "UniversalResource.h"

#include <memory>
#include <cuda_runtime.h>

#include "BitUtil.h"
#include "Util.h"

namespace elsa::mr
{
    namespace universal_resource
    {
        // minimal alignment of pointers returned by any of the cuda malloc calls
        // according to the cuda C programming guide, section 5.3.2 version 12.1
        constexpr size_t GUARANTEED_ALIGNMENT = 256;

        struct ChunkHeader {
            uintptr_t offset;
        };
    } // namespace universal_resource

    MemoryResource UniversalResource::make()
    {
        return MemoryResource::MakeRef(new UniversalResource());
    }

    void* UniversalResource::allocate(size_t size, size_t alignment)
    {
        if (unlikely(!alignment || (alignment & (alignment - 1)))) {
            // alignment is not a power of 2
            throw std::bad_alloc();
        }
        if (size == 0)
            size = 1;
        if (alignment > universal_resource::GUARANTEED_ALIGNMENT) {
            size_t sizeWithAlignment = size + alignment;
            size_t totalSize = sizeWithAlignment + sizeof(universal_resource::ChunkHeader);
            if (unlikely(sizeWithAlignment < size || totalSize < sizeWithAlignment)) {
                throw std::bad_alloc();
            }
            uintptr_t ptr;
            if (unlikely(cudaMallocManaged(reinterpret_cast<void**>(&ptr), size))) {
                throw std::bad_alloc();
            }
            uintptr_t retPtr = alignDown(ptr + totalSize - 1, alignment);
            universal_resource::ChunkHeader* hdr =
                reinterpret_cast<universal_resource::ChunkHeader*>(
                    retPtr - sizeof(universal_resource::ChunkHeader));
            hdr->offset = retPtr - ptr;

            return reinterpret_cast<void*>(retPtr);
        } else {
            void* ptr;
            if (unlikely(cudaMallocManaged(&ptr, size))) {
                throw std::bad_alloc();
            }
            return ptr;
        }
    }

    void UniversalResource::deallocate(void* ptr, size_t size, size_t alignment) noexcept
    {
        static_cast<void>(size);
        if (alignment > universal_resource::GUARANTEED_ALIGNMENT) {
            uintptr_t ptrInt = reinterpret_cast<uintptr_t>(ptr);
            universal_resource::ChunkHeader* hdr =
                reinterpret_cast<universal_resource::ChunkHeader*>(
                    ptrInt - sizeof(universal_resource::ChunkHeader));
            void* allocatedPtr = reinterpret_cast<void*>(ptrInt - hdr->offset);
            cudaFree(allocatedPtr);
        } else {
            cudaFree(ptr);
        }
    }

    bool UniversalResource::tryResize(void* ptr, size_t size, size_t alignment, size_t newSize)
    {
        static_cast<void>(ptr);
        static_cast<void>(size);
        static_cast<void>(alignment);
        static_cast<void>(newSize);
        return false;
    }

} // namespace elsa::mr
#endif
