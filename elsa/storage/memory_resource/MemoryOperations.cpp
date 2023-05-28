#include "MemoryOperations.h"

#include <cstring>

#ifdef ELSA_CUDA_ENABLED
#include <cuda_runtime.h>
static constexpr bool CudaAvailable = true;
#else
static constexpr bool CudaAvailable = false;

/* placeholders */
void cudaMemcpy(void*, const void*, size_t, int) {}
void cudaMemset(void*, int, size_t) {}
void cudaMemPrefetchAsync(const void *, size_t, int) {}
static constexpr int cudaMemcpyDefault = 0;
static constexpr int cudaMemcpyDeviceToDevice = 0;
static constexpr int cudaCpuDeviceId = 0;
#endif

template <class Type>
void fillTyped(void* ptr, const void* src, std::size_t stride, std::size_t totalWrites)
{
    const Type* s = static_cast<const Type*>(src);
    const Type* e = s + stride;
    Type* p = static_cast<Type*>(ptr);

    for (size_t i = 0; i < totalWrites; i++) {
        *p = *s;
        ++p;
        if (++s == e)
            s = static_cast<const Type*>(src);
    }
}

/* expects both pointers to be universally accessible */
void elsa::mr::detail::memOpMove(void* ptr, const void* src, std::size_t size)
{
    std::memmove(ptr, src, size);
}

/* expects ptr to be universally accessible and src to be optional */
void elsa::mr::detail::memOpCopy(void* ptr, const void* src, std::size_t size, bool src_universal)
{
    if (CudaAvailable && src_universal) {
        cudaMemPrefetchAsync(src, size, 0);
        cudaMemPrefetchAsync(ptr, size, 0);
        cudaMemcpy(ptr, src, size, cudaMemcpyDeviceToDevice);
    }
    else {
        if(CudaAvailable) {
            cudaMemPrefetchAsync(src, size, cudaCpuDeviceId);
            cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId);
        }
        std::memcpy(ptr, src, size);
    }
}

/* expects ptr to be universally accessible */
void elsa::mr::detail::memOpSet(void* ptr, const void* src, std::size_t count, std::size_t stride)
{
    size_t equal = 1;
    const uint8_t* p8 = static_cast<const uint8_t*>(src);
    while (equal < stride && p8[0] == p8[equal])
        ++equal;

    if (equal >= stride) {
        if (CudaAvailable)
            cudaMemset(ptr, p8[0], count * stride);
        else
            std::memset(ptr, p8[0], count * stride);
        return;
    }

    if ((stride % 8) == 0)
        fillTyped<uint64_t>(ptr, src, (stride / 8), (stride / 8) * count);
    else if ((stride % 4) == 0)
        fillTyped<uint32_t>(ptr, src, (stride / 4), (stride / 4) * count);
    else if ((stride % 2) == 0)
        fillTyped<uint16_t>(ptr, src, (stride / 2), (stride / 2) * count);
    else
        fillTyped<uint8_t>(ptr, src, stride, stride * count);
}