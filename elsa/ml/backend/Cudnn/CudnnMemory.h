#pragma once

#include <vector>
#include <numeric>
#include <memory>

#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "CudnnCommon.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t, bool isFilter = false>
            class CudnnMemory;

            template <typename data_t>
            class Memory
            {
            public:
                template <typename T, bool B>
                friend class CudnnMemory;

                /// Get pointer to underlying raw memory.
                __host__ __device__ data_t* getMemoryHandle();

                /// Get constant pointer to underlying raw memory.
                __host__ __device__ const data_t* getMemoryHandle() const;

                /// Get size of memory, i.e. the number of elements it holds.
                __host__ __device__ std::size_t getSize() const;

                /// Get size of memory in bytes.
                __host__ __device__ std::size_t getSizeInBytes() const;

                /// True if this memory could have been modified, false
                /// otherwise.
                bool couldBeModified() const;

                virtual void fill(data_t value) = 0;

            protected:
                Memory();

                Memory(std::size_t size, data_t* raw);

                std::size_t size_;
                data_t* raw_;
                mutable bool couldBeModified_ = true;
            };

            template <typename data_t>
            class HostMemory : public Memory<data_t>
            {
            public:
                template <typename T>
                friend void swap(HostMemory<T>& first, HostMemory<T>& second);

                HostMemory() = default;

                HostMemory(std::size_t size);

                HostMemory(const HostMemory& other);

                HostMemory& operator=(HostMemory other);

                HostMemory(HostMemory&& other);

                ~HostMemory();

                void fill(data_t value) override;
            };

            template <typename data_t>
            class DeviceMemory : public Memory<data_t>
            {
            public:
                /// Default constructor
                DeviceMemory() = default;

                /// Construct device memory by specifying its size
                DeviceMemory(std::size_t size);

                /// Copy constructor is deleted since any copy of device memory
                /// would happen via host memory and is therefore way too
                /// expensive
                DeviceMemory(const DeviceMemory&) = delete;

                /// Destruct device memory
                ~DeviceMemory();

                void fill(data_t value) override;
            };

            template <typename CudnnMemoryType>
            struct CudnnDescriptorHelper;

            template <typename data_t>
            struct CudnnDescriptorHelper<CudnnMemory<data_t, false>> {
                using CudnnMemoryType = CudnnMemory<data_t, false>;
                static void construct(CudnnMemoryType* memory)
                {
                    assert(memory->dimensions_.size() == 4
                           && "Dimensions of Cudnn Memory must have size 4");
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                        cudnnCreateTensorDescriptor(&memory->cudnnDescriptor_));
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetTensor4dDescriptor(
                        memory->cudnnDescriptor_, CUDNN_TENSOR_NCHW, memory->typeTag_,
                        memory->dimensions_[0], memory->dimensions_[1], memory->dimensions_[2],
                        memory->dimensions_[3]));
                }

                static void destruct(CudnnMemoryType* memory)
                {
                    if (memory->cudnnDescriptor_) {
                        ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                            cudnnDestroyTensorDescriptor(memory->cudnnDescriptor_));
                    }
                }
            };

            template <typename data_t>
            struct CudnnDescriptorHelper<CudnnMemory<data_t, true>> {
                using CudnnMemoryType = CudnnMemory<data_t, true>;
                static void construct(CudnnMemoryType* memory)
                {
                    cudnnCreateFilterDescriptor(&memory->cudnnDescriptor_);
                }

                static void destruct(CudnnMemoryType* memory)
                {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                        cudnnDestroyFilterDescriptor(memory->cudnnDescriptor_));
                }
            };

            template <typename data_t, bool isFilter>
            class CudnnMemory
            {
            public:
                using CudnnDescriptorType =
                    std::conditional_t<isFilter, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t>;

                friend struct CudnnDescriptorHelper<CudnnMemory>;

                CudnnMemory();

                explicit CudnnMemory(const VolumeDescriptor& descriptor);

                CudnnMemory(const CudnnMemory& other);

                ~CudnnMemory();

                template <typename T, bool B>
                friend void swap(CudnnMemory<T, B>& first, CudnnMemory<T, B>& second);

                CudnnMemory& operator=(CudnnMemory other);

                CudnnMemory(CudnnMemory&& other);

                void allocateDeviceMemory();

                void allocateHostMemory();

                void copyToDevice();

                void copyToHost();

                CudnnDescriptorType& getCudnnDescriptor();

                const CudnnDescriptorType& getCudnnDescriptor() const;

                const std::vector<index_t>& getDimensions() const;

                std::shared_ptr<HostMemory<data_t>> hostMemory;
                std::shared_ptr<DeviceMemory<data_t>> deviceMemory;

            private:
                void constructCudnnDescriptor();

                void destructCudnnDescriptor();

                std::vector<index_t> dimensions_;
                CudnnDescriptorType cudnnDescriptor_;
                cudnnDataType_t typeTag_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa