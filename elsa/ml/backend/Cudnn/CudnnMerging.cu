#include "CudnnMerging.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnMerging<data_t>::CudnnMerging(
                const std::vector<VolumeDescriptor>& inputDescriptors,
                const VolumeDescriptor& outputDescriptor)
                : CudnnLayer<data_t>(inputDescriptors, outputDescriptor, "CudnnMerging",
                                     CudnnLayer<data_t>::anyNumberOfInputs)
            {
            }

            template <typename data_t>
            bool CudnnMerging<data_t>::needsForwardSynchronisation() const
            {
                return true;
            }

            template <typename data_t>
            bool CudnnMerging<data_t>::canMerge() const
            {
                return true;
            }

            template <typename data_t>
            CudnnSum<data_t>::CudnnSum(const std::vector<VolumeDescriptor>& inputDescriptors,
                                       const VolumeDescriptor& outputDescriptor)
                : CudnnMerging<data_t>(inputDescriptors, outputDescriptor)
            {
            }

            template <typename data_t>
            void CudnnSum<data_t>::forwardPropagate()
            {
                BaseType::compileForwardStream();

                output_.deviceMemory->fill(data_t(0));

#pragma omp parallel
                for (auto&& input : input_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* contant 1.f */ &CudnnContext::One,
                        /* i-th input desc */ input.getCudnnDescriptor(),
                        /* i-th input memory */
                        input.deviceMemory->getMemoryHandle(),
                        /* constant 0.f */ &CudnnContext::One,
                        /* output descriptor */ output_.getCudnnDescriptor(),
                        /* output memory  */
                        output_.deviceMemory->getMemoryHandle()));
                }
            }

            // Computes the coefficient-wise product out = in0 * in1
            template <typename data_t>
            __global__ static void coeffwiseMultKernel(std::size_t size, const data_t* in0,
                                                       const data_t* in1, data_t* out)
            {
                ELSA_CUDA_KERNEL_LOOP(index, size) { out[index] = in0[index] * in1[index]; }
            }

            template <typename data_t>
            void CudnnSum<data_t>::backwardPropagate()
            {
                BaseType::compileBackwardStream();

#pragma omp parallel
                // inputGradient[i] = outputGradient * input[i]
                for (std::size_t idx = 0; idx < inputGradient_.size(); ++idx) {
                    const std::size_t size = inputGradient_[idx].deviceMemory->getSize();
                    coeffwiseMultKernel<<<ELSA_CUDA_GET_BLOCKS(size), ELSA_CUDA_NUM_THREADS>>>(
                        size, outputGradient_.front().deviceMemory->getMemoryHandle(),
                        input_[idx].deviceMemory->getMemoryHandle(),
                        inputGradient_[idx].deviceMemory->getMemoryHandle());
                }
            }

            template <typename data_t>
            CudnnConcatenate<data_t>::CudnnConcatenate(
                const std::vector<VolumeDescriptor>& inputDescriptors,
                const VolumeDescriptor& outputDescriptor)
                : CudnnMerging<data_t>(inputDescriptors, outputDescriptor)
            {
            }

            template <typename data_t>
            void CudnnConcatenate<data_t>::forwardPropagate()
            {
            }

            template <typename data_t>
            void CudnnConcatenate<data_t>::backwardPropagate()
            {
            }

            template class CudnnMerging<float>;
            template class CudnnSum<float>;

        } // namespace detail
    }     // namespace ml
} // namespace elsa
