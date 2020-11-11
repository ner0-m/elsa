#pragma once

#include "Logger.h"
#include "elsaDefinesCUDA.cuh"

#include <cuda_runtime.h>

/**
 * \brief Custom macro to check CUDA API calls for errors with line information
 */
#ifndef gpuErrchk
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        if (abort) {
            elsa::Logger::get("CUDA")->critical("{} {}:{}", cudaGetErrorString(code), file, line);
            exit(code);
        } else {
            elsa::Logger::get("CUDA")->error("{} {}:{}", cudaGetErrorString(code), file, line);
        }
    }
}
#endif

#ifndef gpuErrchkNoAbort
#define gpuErrchkNoAbort(ans)                        \
    {                                                \
        gpuAssert((ans), __FILE__, __LINE__, false); \
    }
#endif

namespace elsa
{
    class CudaObjectWrapper
    {
    public:
        CUDA_HOST CudaObjectWrapper() {}

        CUDA_HOST virtual ~CudaObjectWrapper() = default;

        CUDA_HOST CudaObjectWrapper(const CudaObjectWrapper&) = delete;

        CUDA_HOST CudaObjectWrapper& operator=(const CudaObjectWrapper&) = delete;
    };

    template <typename data_t>
    class PinnedArray : public CudaObjectWrapper
    {
    public:
        CUDA_HOST PinnedArray(index_t size)
            : _ptr{allocate(size), [](data_t* ptr) { gpuErrchkNoAbort(cudaFreeHost((void*) ptr)); }}
        {
        }

        CUDA_HOST PinnedArray(PinnedArray<data_t>&& other) : _ptr(std::move(other._ptr)) {}

        CUDA_HOST ~PinnedArray() override = default;

        CUDA_HOST data_t* get() const noexcept { return _ptr.get(); }

        CUDA_HOST data_t& operator[](index_t i) const { return _ptr[i]; }

    private:
        CUDA_HOST data_t* allocate(index_t size)
        {
            data_t* pinnedMemory;
            gpuErrchk(cudaMallocHost((void**) &pinnedMemory, size * sizeof(data_t)));
            return pinnedMemory;
        }

        std::unique_ptr<data_t[], void (*)(data_t*)> _ptr;
    };

    class CudaArrayWrapper : public CudaObjectWrapper
    {
    public:
        CUDA_HOST CudaArrayWrapper(const cudaChannelFormatDesc& channelFormatDesc,
                                   IndexVector_t size, unsigned int flags = 0)
            : arr{allocate(channelFormatDesc, size, flags),
                  [](cudaArray_t ptr) { gpuErrchkNoAbort(cudaFreeArray(ptr)); }}

        {
        }

        CUDA_HOST CudaArrayWrapper(CudaArrayWrapper&& other) : arr(std::move(other.arr)) {}

        CUDA_HOST operator cudaArray_t() const { return arr.get(); }

    private:
        CUDA_HOST cudaArray_t allocate(const cudaChannelFormatDesc& channelFormatDesc,
                                       IndexVector_t size, unsigned int flags)
        {
            auto sizeui = size.template cast<size_t>();

            cudaArray_t tmp;
            switch (sizeui.size()) {
                case 1:
                    gpuErrchk(cudaMallocArray(&tmp, &channelFormatDesc, sizeui[0], 0, flags));
                    break;
                case 2:
                    if (flags == cudaArrayLayered) {
                        gpuErrchk(cudaMalloc3DArray(&tmp, &channelFormatDesc,
                                                    make_cudaExtent(sizeui[0], 0, sizeui[1]),
                                                    flags));
                    } else {
                        gpuErrchk(
                            cudaMallocArray(&tmp, &channelFormatDesc, sizeui[0], sizeui[1], flags));
                    }
                    break;
                case 3:
                    gpuErrchk(cudaMalloc3DArray(&tmp, &channelFormatDesc,
                                                make_cudaExtent(sizeui[0], sizeui[1], sizeui[2]),
                                                flags));
                    break;
                default:
                    throw std::invalid_argument("cudaArray cannot have more than 3 dimensions");
            }

            return tmp;
        }

        std::unique_ptr<cudaArray, void (*)(cudaArray_t)> arr;
    };

    class TextureWrapper : public CudaObjectWrapper
    {
    public:
        CUDA_HOST TextureWrapper(cudaArray_t array, const cudaTextureDesc& textureDesc,
                                 const cudaResourceViewDesc* resourceViewDesc = nullptr)
            : texture{allocate(array, textureDesc, resourceViewDesc), [](cudaTextureObject_t* ptr) {
                          gpuErrchkNoAbort(cudaDestroyTextureObject(*ptr));
                          delete ptr;
                      }}
        {
        }

        CUDA_HOST TextureWrapper(TextureWrapper&& other) : texture(std::move(other.texture)) {}

        CUDA_HOST operator cudaTextureObject_t() const { return *texture; }

    private:
        cudaTextureObject_t* allocate(cudaArray_t array, const cudaTextureDesc& textureDesc,
                                      const cudaResourceViewDesc* resourceViewDesc)
        {
            cudaTextureObject_t* tmp = new cudaTextureObject_t();

            cudaResourceDesc resourceDesc;
            std::memset(&resourceDesc, 0, sizeof(cudaResourceDesc));
            resourceDesc.res.array = {array};
            resourceDesc.resType = cudaResourceTypeArray;

            gpuErrchk(cudaCreateTextureObject(tmp, &resourceDesc, &textureDesc, resourceViewDesc));

            return tmp;
        }

        std::unique_ptr<cudaTextureObject_t, void (*)(cudaTextureObject_t*)> texture;
    };

    template <typename data_t>
    class PitchedPtrWrapper : public CudaObjectWrapper
    {
    public:
        CUDA_HOST PitchedPtrWrapper(IndexVector_t size)
            : pitchedPtr{allocate(size), [](cudaPitchedPtr* ptr) {
                             gpuErrchkNoAbort(cudaFree(ptr->ptr));
                             delete ptr;
                         }}
        {
        }

        CUDA_HOST PitchedPtrWrapper(PitchedPtrWrapper<data_t>&& other)
            : pitchedPtr(std::move(other.pitchedPtr))
        {
        }

        CUDA_HOST operator cudaPitchedPtr&() { return *pitchedPtr; };

    private:
        cudaPitchedPtr* allocate(IndexVector_t size)
        {
            auto tmp = new cudaPitchedPtr();

            auto sizeui = size.template cast<size_t>();

            switch (sizeui.size()) {
                case 2:
                    gpuErrchk(cudaMallocPitch(&tmp->ptr, &tmp->pitch, sizeui[0] * sizeof(data_t),
                                              sizeui[1]));
                    break;
                case 3:
                    gpuErrchk(cudaMalloc3D(
                        tmp, make_cudaExtent(sizeui[0] * sizeof(data_t), sizeui[1], sizeui[2])));
                    break;
                default:
                    throw std::invalid_argument(
                        "cudaPitchedPtr cannot must be 2- or 3-dimensional");
            }

            return tmp;
        }

        std::unique_ptr<cudaPitchedPtr, void (*)(cudaPitchedPtr*)> pitchedPtr;
    };

    class CudaStreamWrapper : public CudaObjectWrapper
    {
    public:
        CUDA_HOST CudaStreamWrapper()
            : stream{allocate(), [](cudaStream_t ptr) { gpuErrchkNoAbort(cudaStreamDestroy(ptr)); }}
        {
        }

        CUDA_HOST CudaStreamWrapper(CudaStreamWrapper&& other) : stream(std::move(other.stream)) {}

        CUDA_HOST operator cudaStream_t() const { return stream.get(); }

    private:
        cudaStream_t allocate()
        {
            cudaStream_t tmp;
            gpuErrchk(cudaStreamCreate(&tmp));
            return tmp;
        }

        std::unique_ptr<CUstream_st, void (*)(cudaStream_t)> stream;
    };
} // namespace elsa