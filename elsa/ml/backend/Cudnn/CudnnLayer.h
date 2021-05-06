#pragma once

#include <vector>
#include <memory>
#include <algorithm>

// #include "Common.h"
#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "CudnnContext.h"
#include "CudnnMemory.h"
#include "Common.h"

#include "cublas_v2.h"
#include "cudnn.h"

namespace elsa
{
    template <typename data_t>
    class DataContainer;

    namespace ml
    {
        namespace detail
        {
            // nvcc does not support C++17 which is required for the use of
            // DataContainer. Since we want to input DataContainers into
            // CudnnLayers we use the following small trick to make it work:
            //
            // C++14 and C++17 are ABI compatible, we therefore postpone the
            // point at which CudnnLayer needs to know the implementation of
            // DataContainer to link-time. At compile-time we just forward
            // declare DataContainer and a CudnnDataContainerInterface which is
            // located in the ml module rather than in the backend module and
            // can hence use C++17 features (such as including DataConainer).
            //
            // Note that this implies that CudnnLayer has to be header-only.
            template <typename data_t>
            struct CudnnDataContainerInterface;

            // /// Reverse a volume-descriptor
            // ///
            // /// If we have a descriptor
            // ///   {w, h, c, n}
            // /// this creates a descriptor
            // ///   {n, c, h, w}.
            // static inline VolumeDescriptor reverseVolumeDescriptor(const VolumeDescriptor& desc)
            // {
            //     IndexVector_t dims = desc.getNumberOfCoefficientsPerDimension().reverse();
            //     return VolumeDescriptor(dims);
            // }

            /// Base class for all Cudnn layers
            ///
            /// @author David Tellenbach
            template <typename data_t>
            class CudnnLayer
            {
            public:
                /// Set the CudnnContext for this layer.
                void setCudnnContext(std::shared_ptr<CudnnContext> cudnnContext);

                /// Forward propagate this layer's input
                virtual void forwardPropagate();

                /// Backward propagate this layer's output-gradient
                virtual void backwardPropagate();

                /// Set this layer's input at a given index
                void setInput(const DataContainer<data_t>& input, index_t index = 0);

                /// @returns a DataContainer with this layers output.
                DataContainer<data_t> getOutput();

                /// Set this layer's input memory at a given index
                void setInputMemory(std::shared_ptr<DeviceMemory<data_t>> inputMemory,
                                    index_t index = 0);

                /// Set this layer's next input memory
                void setNextInputMemory(std::shared_ptr<DeviceMemory<data_t>> inputMemory);

                /// Set this output-gradient memory at a given index
                void setOutputGradientMemory(
                    std::shared_ptr<DeviceMemory<data_t>> outputGradientMemory, index_t index = 0);

                void setNextOutputGradientMemory(
                    std::shared_ptr<DeviceMemory<data_t>> outputGradientMemory);

                void setOutputGradient(const DataContainer<data_t>& outputGradient,
                                       index_t index = 0);

                std::shared_ptr<DeviceMemory<data_t>> getInputMemory()
                {
                    return input_.front().deviceMemory;
                }

                std::shared_ptr<DeviceMemory<data_t>> getOutputGradientMemory()
                {
                    return outputGradient_.front().deviceMemory;
                }

                /// @returns this layer's input-gradient at a given index
                DataContainer<data_t> getInputGradient(index_t index = 0);

                /// @returns the number of inputs of this layer
                index_t getNumberOfInputs() const;

                std::shared_ptr<DeviceMemory<data_t>> getOutputMemory();

                std::shared_ptr<DeviceMemory<data_t>> getOutputMemory() const;

                /// @returns this layer's input-descriptor at a given index
                VolumeDescriptor getInputDescriptor(index_t index = 0) const;

                /// @returns this layer's output-descriptor
                VolumeDescriptor getOutputDescriptor() const;

                /// @returns true if this layer has trainable parameters, false otherwise
                virtual bool isTrainable() const;

                /// @returns true if this layer can merge multiple inputs, false otherwise.
                virtual bool canMerge() const;

                /// @returns true if this layer needs to block the execution of subsequent layers
                /// during backward propagation, false otherwise
                virtual bool needsForwardSynchronisation() const;

                /// @returns true if this layer needs to block the execution of previous layers
                /// during backward propagation, false otherwise.
                virtual bool needsBackwardSynchronisation() const;

                /// @returns this layer's name
                std::string getName() const;

                /// Set the number of output-gradients of this layer.
                void setNumberOfOutputGradients(index_t num);

                /// Get this layer's input-gradient memory at a given index.
                std::shared_ptr<DeviceMemory<data_t>> getInputGradientMemory(index_t index = 0);

                /// Prepare this layer for forward propagation.
                virtual void compileForwardStream();

                /// Prepare this layer for backward propagation.
                virtual void compileBackwardStream();

                /// Handle multiple output gradients by replacing the first
                /// output-gradient of this layer by the sum of all output-gradients
                void handleMultipleOutputGradients();

                /// @returns the batch-size used in this layer
                index_t getBatchSize() const;

            protected:
                /// Construct a CudnnLayer by specifying a list of input- and a single
                /// output-descriptors.
                CudnnLayer(const std::vector<VolumeDescriptor>& inputDescriptor,
                           const VolumeDescriptor& outputDescriptor, const std::string& name,
                           index_t allowedNumberOfInputs = 1);

                /// Construct a CudnnLayer by specifying a single input- and a single
                /// output-descriptor
                CudnnLayer(const VolumeDescriptor& inputDescriptor,
                           const VolumeDescriptor& outputDescriptor, const std::string& name,
                           index_t allowedNumberOfInputs = 1);

                /// Validata an access to an index of a std::vector
                template <typename T>
                inline static void validateVectorIndex([[maybe_unused]] const std::vector<T>& vec,
                                                       [[maybe_unused]] index_t index);

                /// Validate device-memory behind a given CudnnMemory pointer.
                template <bool isFilter>
                static void validateDeviceMemory(const CudnnMemory<data_t, isFilter>& mem);

                /// Validate host-memory behind a given CudnnMemory pointer.
                template <bool isFilter>
                static void validateHostMemory(const CudnnMemory<data_t, isFilter>& mem);

                /// Validate the correctness of data used during forward propagation of this layer.
                virtual void validateForwardPropagation();

                /// Validate the correctness of data used during backward propagation of this layer.
                virtual void validateBackwardPropagation();

                /// Any number of inputs is allowed, e.g., for merging layers.
                static constexpr index_t anyNumberOfInputs = -1;

                /// Batch-size in this layer.
                const index_t batchSize_;

                /// This layer's Cudnn context.
                std::shared_ptr<CudnnContext> cudnnContext_;

                /// A list of this layer's input-descriptors.
                std::vector<std::unique_ptr<DataDescriptor>> inputDescriptor_;

                /// This layer's output-descriptor.
                std::unique_ptr<DataDescriptor> outputDescriptor_;

                /// A list of this layer's inputs.
                std::vector<CudnnMemory<data_t>> input_;

                /// This layer's output
                CudnnMemory<data_t> output_;

                /// A list of this layer's input gradients
                std::vector<CudnnMemory<data_t>> inputGradient_;

                /// A list of this layer's output gradients
                std::vector<CudnnMemory<data_t>> outputGradient_;

                /// The name of this layer.
                std::string name_;

                /// The number of allowed inputs of this layer
                index_t allowedNumberOfInputs_ = 1;

                /// Flag to indicate whether this layer has been compiled for backward propagation.
                bool isBackwardCompiled_ = false;

                /// Flag to indicate whether this layer has been compiled for forward propagation.
                bool isForwardCompiled_ = false;

            private:
                index_t currentInputMemoryIndex_ = 0;
                index_t currentOutputGradientMemoryIndex_ = 0;
            };

            template <typename data_t>
            void CudnnLayer<data_t>::setCudnnContext(std::shared_ptr<CudnnContext> cudnnContext)
            {
                cudnnContext_ = cudnnContext;
            }

            template <typename data_t>
            void CudnnLayer<data_t>::forwardPropagate()
            {
                Logger::get(getName())->trace("Forward propagate");
            }

            template <typename data_t>
            void CudnnLayer<data_t>::backwardPropagate()
            {
                Logger::get(getName())->trace("Backward propagate");
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setInput(const DataContainer<data_t>& input, index_t index)
            {
                validateVectorIndex(input_, index);
                CudnnDataContainerInterface<data_t>::getCudnnMemoryFromDataContainer(
                    input.viewAs(getInputDescriptor(index)), &input_[index]);
                input_[index].copyToDevice();
            }

            template <typename data_t>
            DataContainer<data_t> CudnnLayer<data_t>::getOutput()
            {
                VolumeDescriptor desc(
                    getOutputDescriptor().getNumberOfCoefficientsPerDimension().reverse());
                output_.copyToHost();
                return CudnnDataContainerInterface<data_t>::getDataContainerFromCudnnMemory(output_,
                                                                                            desc);
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setInputMemory(
                std::shared_ptr<DeviceMemory<data_t>> inputMemory, index_t index)
            {
                validateVectorIndex(input_, index);
                input_[index].deviceMemory = inputMemory;
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setNextInputMemory(
                std::shared_ptr<DeviceMemory<data_t>> inputMemory)
            {
                index_t nextIndex = currentInputMemoryIndex_++;
                setInputMemory(inputMemory, nextIndex);
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setOutputGradientMemory(
                std::shared_ptr<DeviceMemory<data_t>> outputGradientMemory, index_t index)
            {
                validateVectorIndex(outputGradient_, index);
                outputGradient_[index].deviceMemory = outputGradientMemory;
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setNextOutputGradientMemory(
                std::shared_ptr<DeviceMemory<data_t>> outputGradientMemory)
            {
                index_t nextIndex = currentOutputGradientMemoryIndex_++;
                setOutputGradientMemory(outputGradientMemory, nextIndex);
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setOutputGradient(const DataContainer<data_t>& outputGradient,
                                                       index_t index)
            {
                validateVectorIndex(outputGradient_, index);

                // Get CudnnMemory from DataContainer
                CudnnDataContainerInterface<data_t>::getCudnnMemoryFromDataContainer(
                    outputGradient, &outputGradient_[index]);
                outputGradient_.back().copyToDevice();
            }

            template <typename data_t>
            DataContainer<data_t> CudnnLayer<data_t>::getInputGradient(index_t index)
            {
                validateVectorIndex(inputGradient_, index);
                inputGradient_[index].copyToHost();
                return CudnnDataContainerInterface<data_t>::getDataContainerFromCudnnMemory(
                    inputGradient_[index]);
            }

            template <typename data_t>
            std::shared_ptr<DeviceMemory<data_t>>
                CudnnLayer<data_t>::getInputGradientMemory(index_t index)
            {
                validateVectorIndex(inputGradient_, index);
                return inputGradient_[index].deviceMemory;
            }

            template <typename data_t>
            std::shared_ptr<DeviceMemory<data_t>> CudnnLayer<data_t>::getOutputMemory()
            {
                return output_.deviceMemory;
            }

            template <typename data_t>
            VolumeDescriptor CudnnLayer<data_t>::getInputDescriptor(index_t index) const
            {
                validateVectorIndex(inputDescriptor_, index);
                return *dynamic_unique_ptr_cast<VolumeDescriptor>(inputDescriptor_[index]->clone());
            }

            template <typename data_t>
            VolumeDescriptor CudnnLayer<data_t>::getOutputDescriptor() const
            {
                return *dynamic_unique_ptr_cast<VolumeDescriptor>(outputDescriptor_->clone());
            }

            template <typename data_t>
            CudnnLayer<data_t>::CudnnLayer(const std::vector<VolumeDescriptor>& inputDescriptor,
                                           const VolumeDescriptor& outputDescriptor,
                                           const std::string& name, index_t allowedNumberOfInputs)
                : outputDescriptor_(outputDescriptor.clone()),
                  name_(name),
                  allowedNumberOfInputs_(allowedNumberOfInputs),
                  output_(outputDescriptor),
                  batchSize_(inputDescriptor.front().getNumberOfCoefficientsPerDimension()[0])
            {
                assert(inputDescriptor.size() > 0
                       && "CudnnLayer needs at least one input-descriptor");

                // Check that all input-descriptors have the same batch-size
                if (std::adjacent_find(inputDescriptor.begin(), inputDescriptor.end(),
                                       [](const auto& a, const auto& b) {
                                           return a.getNumberOfCoefficientsPerDimension()[0]
                                                  != b.getNumberOfCoefficientsPerDimension()[0];
                                       })
                    != inputDescriptor.end()) {
                    assert(false && "Batch-size must be the same for all input-descriptors");
                }

                // Set input-decriptor, input-memory and input-gradient-memory
                for (std::size_t i = 0; i < inputDescriptor.size(); ++i) {
                    inputDescriptor_.push_back(inputDescriptor[i].clone());
                    input_.emplace_back(inputDescriptor[i]);
                    inputGradient_.emplace_back(inputDescriptor[i]);

                    // Allocate input-gradient device memory
                    inputGradient_[i].allocateDeviceMemory();
                }

                // Set default Cudnn context
                cudnnContext_ = std::make_shared<CudnnContext>();

                // Allocate output device-memory
                output_.allocateDeviceMemory();
            }

            template <typename data_t>
            CudnnLayer<data_t>::CudnnLayer(const VolumeDescriptor& inputDescriptor,
                                           const VolumeDescriptor& outputDescriptor,
                                           const std::string& name, index_t allowedNumberOfInputs)
                : CudnnLayer<data_t>(std::vector<VolumeDescriptor>({inputDescriptor}),
                                     outputDescriptor, name, allowedNumberOfInputs)
            {
            }

            template <typename data_t>
            template <typename T>
            void CudnnLayer<data_t>::validateVectorIndex([[maybe_unused]] const std::vector<T>& vec,
                                                         [[maybe_unused]] index_t index)
            {
                assert(index >= 0 && asIndex(index) < vec.size()
                       && "Vector index is out of bounds");
            }

            template <typename data_t>
            template <bool isFilter>
            void CudnnLayer<data_t>::validateDeviceMemory(const CudnnMemory<data_t, isFilter>& mem)
            {
                assert(mem.deviceMemory != nullptr && "Device memory cannot be null");
                assert(mem.deviceMemory->getSize() != 0 && "Size of device memory cannot be null");
            }

            template <typename data_t>
            template <bool isFilter>
            void CudnnLayer<data_t>::validateHostMemory(const CudnnMemory<data_t, isFilter>& mem)
            {
                assert(mem.hostMemory != nullptr && "Host memory cannot be null");
                assert(mem.hostMemory->getSize() != 0 && "Size of host memory cannot be null");
            }

            template <typename data_t>
            void CudnnLayer<data_t>::validateForwardPropagation()
            {
                assert(isForwardCompiled_
                       && "Cannot forward propagate since layer has not been compiled");
                for (auto&& input : input_) {
                    validateDeviceMemory(input);
                }

                validateDeviceMemory(output_);
            }

            template <typename data_t>
            void CudnnLayer<data_t>::validateBackwardPropagation()
            {
                // Check that layer has been compile for both, forward and
                // backward propagation
                assert(isForwardCompiled_ && isBackwardCompiled_
                       && "Cannot backward propagate since layer has not been compiled");

                // Check that we have at least one output-gradient
                assert(outputGradient_.size() != 0
                       && "Cannot backward propagate since no output-gradient has been set");

                for (auto&& outGrad : outputGradient_) {
                    validateDeviceMemory(outGrad);
                }

                for (auto&& inGrad : inputGradient_) {
                    validateDeviceMemory(inGrad);
                }

                handleMultipleOutputGradients();
            }

            template <typename data_t>
            void CudnnLayer<data_t>::compileForwardStream()
            {
                // Check that we don't have too many input for this layer
                assert(input_.size() == allowedNumberOfInputs_
                       || allowedNumberOfInputs_ == anyNumberOfInputs
                              && "Cannot forward propagate because layer has too many inputs");

                // Allocate input device memory and validate it
                for (auto&& input : input_) {
                    input.allocateDeviceMemory();
                    validateDeviceMemory(input);
                }

                // Allocate output memory
                output_.allocateDeviceMemory();

                isForwardCompiled_ = true;
            }

            template <typename data_t>
            void CudnnLayer<data_t>::compileBackwardStream()
            {
                // Allocate output gradient memory  and validate it
                for (auto&& outGrad : outputGradient_) {
                    outGrad.allocateDeviceMemory();
                    validateDeviceMemory(outGrad);
                }
                isBackwardCompiled_ = true;
            }

            template <typename data_t>
            bool CudnnLayer<data_t>::isTrainable() const
            {
                return false;
            }

            template <typename data_t>
            bool CudnnLayer<data_t>::needsForwardSynchronisation() const
            {
                return canMerge();
            }

            template <typename data_t>
            bool CudnnLayer<data_t>::canMerge() const
            {
                return false;
            }

            template <typename data_t>
            bool CudnnLayer<data_t>::needsBackwardSynchronisation() const
            {
                if (outputGradient_.size() > 1)
                    return true;
                return false;
            }

            template <typename data_t>
            std::string CudnnLayer<data_t>::getName() const
            {
                return name_;
            }

            template <typename data_t>
            index_t CudnnLayer<data_t>::getNumberOfInputs() const
            {
                return input_.size();
            }

            template <typename data_t>
            void CudnnLayer<data_t>::setNumberOfOutputGradients(index_t num)
            {
                // We need at least one output gradient
                const index_t numOutputGrads = !num ? 1 : num;
                for (std::size_t i = 0; i < numOutputGrads; ++i) {
                    outputGradient_.emplace_back(getOutputDescriptor());
                }
            }

            template <typename data_t>
            void CudnnLayer<data_t>::handleMultipleOutputGradients()
            {
                if (outputGradient_.size() > 1) {
                    for (auto&& outgrad : outputGradient_) {
                        for (auto&& input : input_) {
                            ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                                /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                                /* contant 1.f */ &CudnnContext::One,
                                /* i-th outgrad desc */ outgrad.getCudnnDescriptor(),
                                /* i-th outgrad memory */
                                outgrad.deviceMemory->getMemoryHandle(),
                                /* constant 0.f */ &CudnnContext::One,
                                /* first outgrad descriptor */
                                outputGradient_.front().getCudnnDescriptor(),
                                /* first outgrad memory  */
                                outputGradient_.front().deviceMemory->getMemoryHandle()));
                        }
                    }
                }
            }

            template <typename data_t>
            index_t CudnnLayer<data_t>::getBatchSize() const
            {
                return batchSize_;
            }
        } // namespace detail
    }     // namespace ml
} // namespace elsa