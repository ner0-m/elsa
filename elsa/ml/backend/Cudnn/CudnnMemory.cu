#include "CudnnMemory.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            Memory<data_t>::Memory() : size_(0), raw_(nullptr)
            {
            }

            template <typename data_t>
            Memory<data_t>::Memory(std::size_t size, data_t* raw) : size_(size), raw_(raw)
            {
            }

            template <typename data_t>
            data_t* Memory<data_t>::getMemoryHandle()
            {
                couldBeModified_ = true;
                return raw_;
            }

            template <typename data_t>
            const data_t* Memory<data_t>::getMemoryHandle() const
            {
                return raw_;
            }

            template <typename data_t>
            std::size_t Memory<data_t>::getSize() const
            {
                return size_;
            }
            template <typename data_t>
            std::size_t Memory<data_t>::getSizeInBytes() const
            {
                return getSize() * sizeof(data_t);
            }

            template <typename data_t>
            bool Memory<data_t>::couldBeModified() const
            {
                return couldBeModified_;
            }

            template class Memory<float>;

            template <typename data_t>
            HostMemory<data_t>::HostMemory(std::size_t size)
                : Memory<data_t>(size, new data_t[size])
            {
            }

            template <typename data_t>
            HostMemory<data_t>::HostMemory(const HostMemory<data_t>& other)
                : Memory<data_t>(other.size_, other.size_ ? new data_t[other.size_] : nullptr)
            {
                std::copy(other.raw_, other.raw_ + this->size_, this->raw_);
            }

            template <typename data_t>
            HostMemory<data_t>::~HostMemory()
            {
                delete[] this->raw_;
            }

            template <typename data_t>
            void swap(HostMemory<data_t>& first, HostMemory<data_t>& second)
            {
                using std::swap;
                swap(first.size_, second.size_);
                swap(first.raw_, second.raw_);
            }

            template <typename data_t>
            HostMemory<data_t>& HostMemory<data_t>::operator=(HostMemory<data_t> other)
            {
                swap(*this, other);
                return *this;
            }

            template <typename data_t>
            HostMemory<data_t>::HostMemory(HostMemory<data_t>&& other)
            {
                swap(*this, other);
            }

            template <typename data_t>
            void HostMemory<data_t>::fill(data_t value)
            {
                assert(this->raw_ != nullptr && "Cannot fill empty memory block");
                std::fill_n(this->raw_, this->getSize(), value);
            }

            template class HostMemory<float>;

            template <typename data_t>
            DeviceMemory<data_t>::DeviceMemory(std::size_t size) : Memory<data_t>(size, nullptr)
            {
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudaMalloc((void**) &this->raw_, this->getSizeInBytes()));
            }

            template <typename data_t>
            DeviceMemory<data_t>::~DeviceMemory()
            {
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudaFree(this->raw_));
            }

            template <typename data_t>
            __global__ static void deviceFill(data_t* vec, std::size_t length, data_t value)
            {
                ELSA_CUDA_KERNEL_LOOP(index, length) { vec[index] = data_t(value); }
            }

            template <typename data_t>
            void DeviceMemory<data_t>::fill(data_t value)
            {
                assert(this->raw_ != nullptr && "Cannot fill empty memory block");
                deviceFill<<<ELSA_CUDA_GET_BLOCKS(this->getSize()), ELSA_CUDA_NUM_THREADS>>>(
                    this->raw_, this->getSize(), value);
            }

            template class DeviceMemory<float>;

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::CudnnMemory()
                : hostMemory(nullptr), deviceMemory(nullptr), cudnnDescriptor_(nullptr)
            {
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::CudnnMemory(const VolumeDescriptor& descriptor)
                : typeTag_(TypeToCudnnTypeTag<data_t>::Value)
            {
                // Cudnn requires a descriptor to have at least 4 dimensions.
                // Since we want to be able to pass VolumeDescriptors with
                // fewer dimensions, we artificially append dimensions of size 1
                assert(descriptor.getNumberOfDimensions() >= 2
                       && descriptor.getNumberOfDimensions() <= 4);

                int dim = 0;
                for (; dim < descriptor.getNumberOfDimensions(); ++dim) {
                    dimensions_.push_back(descriptor.getNumberOfCoefficientsPerDimension()[dim]);
                }

                // Fill to 4 if necessary
                for (; dim < 4; ++dim) {
                    dimensions_.push_back(index_t(1));
                }

                // Construct Cudnn descriptor
                constructCudnnDescriptor();
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::CudnnMemory(const CudnnMemory<data_t, isFilter>& other)
                : dimensions_(other.dimensions_),
                  typeTag_(other.typeTag_),
                  hostMemory(other.hostMemory),
                  deviceMemory(other.deviceMemory)
            {
                // Copy Cudnn descriptor by creating a new one
                constructCudnnDescriptor();
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::~CudnnMemory()
            {
                destructCudnnDescriptor();
            }

            template <typename data_t, bool isFilter>
            void swap(CudnnMemory<data_t, isFilter>& first, CudnnMemory<data_t, isFilter>& second)
            {
                using std::swap;
                swap(first.dimensions_, second.dimensions_);
                swap(first.typeTag_, second.typeTag_);
                swap(first.hostMemory, second.hostMemory);
                swap(first.deviceMemory, second.deviceMemory);
                swap(first.cudnnDescriptor_, second.cudnnDescriptor_);
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>&
                CudnnMemory<data_t, isFilter>::operator=(CudnnMemory<data_t, isFilter> other)
            {
                swap(*this, other);
                return *this;
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::CudnnMemory(CudnnMemory<data_t, isFilter>&& other)
            {
                swap(*this, other);
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::allocateDeviceMemory()
            {
                assert(dimensions_.size() == 4
                       && "Cannot allocate device memory without valid number of dimensions");
                if (!deviceMemory) {
                    deviceMemory = std::make_shared<DeviceMemory<data_t>>(
                        std::accumulate(dimensions_.begin(), dimensions_.end(), std::size_t(1),
                                        std::multiplies<std::size_t>()));
                }
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::allocateHostMemory()
            {
                assert(dimensions_.size() == 4
                       && "Cannot allocate host memory without valid number of dimensions");
                if (!hostMemory) {
                    hostMemory = std::make_shared<HostMemory<data_t>>(
                        std::accumulate(dimensions_.begin(), dimensions_.end(), std::size_t(1),
                                        std::multiplies<std::size_t>()));
                }
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::copyToDevice()
            {
                assert(hostMemory != nullptr && "Cannot copy to device since host-memory is null");
                assert(hostMemory->getSize() != 0
                       && "Cannot copy to device since host-memory has size 0");

                if (!deviceMemory) {
                    allocateDeviceMemory();
                }

                if (hostMemory->couldBeModified()) {
                    cudaMemcpy(deviceMemory->getMemoryHandle(), hostMemory->getMemoryHandle(),
                               hostMemory->getSizeInBytes(), cudaMemcpyHostToDevice);
                    hostMemory->couldBeModified_ = false;
                }
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::copyToHost()
            {
                if (!hostMemory) {
                    allocateHostMemory();
                }

                assert(deviceMemory != nullptr
                       && "Cannot copy to host since device-memory is null");
                assert(deviceMemory->getSize() != 0
                       && "Cannot copy to host since device-memory has size 0");

                if (deviceMemory->couldBeModified()) {
                    cudaMemcpy(hostMemory->getMemoryHandle(), deviceMemory->getMemoryHandle(),
                               deviceMemory->getSizeInBytes(), cudaMemcpyDeviceToHost);
                    deviceMemory->couldBeModified_ = false;
                }
            }

            template <typename data_t, bool isFilter>
            CudnnMemory<data_t, isFilter>::CudnnDescriptorType&
                CudnnMemory<data_t, isFilter>::getCudnnDescriptor()
            {
                return cudnnDescriptor_;
            }

            template <typename data_t, bool isFilter>
            const CudnnMemory<data_t, isFilter>::CudnnDescriptorType&
                CudnnMemory<data_t, isFilter>::getCudnnDescriptor() const
            {
                return cudnnDescriptor_;
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::constructCudnnDescriptor()
            {
                assert(dimensions_.size() == 4
                       && "Cannot create CudnnDescriptor with dimensions not equal to 4");

                CudnnDescriptorHelper<CudnnMemory<data_t, isFilter>>::construct(this);
            }

            template <typename data_t, bool isFilter>
            void CudnnMemory<data_t, isFilter>::destructCudnnDescriptor()
            {
                CudnnDescriptorHelper<CudnnMemory<data_t, isFilter>>::destruct(this);
            }

            template <typename data_t, bool isFilter>
            const std::vector<index_t>& CudnnMemory<data_t, isFilter>::getDimensions() const
            {
                return dimensions_;
            }

            template class CudnnMemory<float, /* isFilter */ true>;
            template class CudnnMemory<float, /* isFilter */ false>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa