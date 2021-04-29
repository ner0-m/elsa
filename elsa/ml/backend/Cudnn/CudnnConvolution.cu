#include "CudnnConvolution.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            CudnnConvolution<data_t>::CudnnConvolution(
                const VolumeDescriptor& inputDescriptor, const VolumeDescriptor& outputDescriptor,
                const VolumeDescriptor& weightsDescriptor, const IndexVector_t& strides,
                const IndexVector_t& paddingLow, const IndexVector_t& paddingHigh, bool useBias,
                Initializer weightsInitializer, Initializer biasInitializer)
                : CudnnTrainable<data_t, false>(inputDescriptor, outputDescriptor,
                                                weightsDescriptor, useBias, weightsInitializer,
                                                biasInitializer)

            {
                // Create convolution descriptor
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreateConvolutionDescriptor(&convolutionDescriptor_));

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetConvolution2dDescriptor(
                    /* conv desc */ convolutionDescriptor_,
                    /* horizontal padding */ paddingLow[0] / 2,
                    /* vertical padding */ paddingHigh[0] / 2,
                    /* vertical stride */ strides[0],
                    /* horizontal stride */ strides[1],
                    /* dilation */ 1, 1,
                    /* conv mode */ CUDNN_CONVOLUTION,
                    /* type tag */ CUDNN_DATA_FLOAT));
            }

            template <typename data_t>
            CudnnConvolution<data_t>::~CudnnConvolution()
            {
                // Destroy conv descriptor
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnDestroyConvolutionDescriptor(convolutionDescriptor_));
            }

            template <typename data_t>
            void CudnnConvolution<data_t>::compileForwardStream()
            {
                BaseType::compileForwardStream();

                // Create filter descriptor for weights
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreateFilterDescriptor(&cudnnFilterDescriptor_));

                // weights are in oihw format
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetFilter4dDescriptor(
                    cudnnFilterDescriptor_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    weights_.getDimensions()[0], weights_.getDimensions()[1],
                    weights_.getDimensions()[2], weights_.getDimensions()[3]));

                std::array<int, 4> output_size_;
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolution2dForwardOutputDim(
                    convolutionDescriptor_, input_.front().getCudnnDescriptor(),
                    cudnnFilterDescriptor_, &output_size_[0], &output_size_[1], &output_size_[2],
                    &output_size_[3]));

                // Since the workspace size we need to allocate may depend on
                // the forward and the backward pass, we choose the algorithm
                // for both here.
                std::size_t wsSize = 0;
                std::size_t tempSize = 0;
                // Choose forward algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionForwardAlgorithm(
                    cudnnContext_->getCudnnHandle(), input_.front().getCudnnDescriptor(),
                    cudnnFilterDescriptor_, convolutionDescriptor_, output_.getCudnnDescriptor(),
                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convolutionForwardAlgorithm_));

                // Get forward workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), input_.front().getCudnnDescriptor(),
                    cudnnFilterDescriptor_, convolutionDescriptor_, output_.getCudnnDescriptor(),
                    convolutionForwardAlgorithm_, &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Choose backward data algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardDataAlgorithm(
                    cudnnContext_->getCudnnHandle(), cudnnFilterDescriptor_,
                    output_.getCudnnDescriptor(), convolutionDescriptor_,
                    input_.front().getCudnnDescriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                    0, &convolutionBackwardDataAlgorithm_));

                // Get backward data workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), cudnnFilterDescriptor_,
                    output_.getCudnnDescriptor(), convolutionDescriptor_,
                    input_.front().getCudnnDescriptor(), convolutionBackwardDataAlgorithm_,
                    &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Choose backward filter algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardFilterAlgorithm(
                    cudnnContext_->getCudnnHandle(), input_.front().getCudnnDescriptor(),
                    output_.getCudnnDescriptor(), convolutionDescriptor_, cudnnFilterDescriptor_,
                    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                    &convolutionBackwardFilterAlgorithm_));

                // Get backward data workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), input_.front().getCudnnDescriptor(),
                    output_.getCudnnDescriptor(), convolutionDescriptor_, cudnnFilterDescriptor_,
                    convolutionBackwardFilterAlgorithm_, &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Allocate workspace memory
                workspaceMemory_ = std::make_shared<DeviceMemory<data_t>>(wsSize);
            }

            template <typename data_t>
            void CudnnConvolution<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionForward(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* filter desc */ cudnnFilterDescriptor_,
                    /* weights memory */ weights_.deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv fwd algo */ convolutionForwardAlgorithm_,
                    /* workspace */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* output desc */ output_.getCudnnDescriptor(),
                    /* output memory */ output_.deviceMemory->getMemoryHandle()));

                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* bias desc */ bias_.getCudnnDescriptor(),
                        /* bias memory */ bias_.deviceMemory->getMemoryHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* output desc */ output_.getCudnnDescriptor(),
                        /* output memory */ output_.deviceMemory->getMemoryHandle()));
                }
            }

            template <typename data_t>
            void CudnnConvolution<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardBias(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* output-grad desc */ outputGradient_.front().getCudnnDescriptor(),
                        /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                        /* constant 0.f */ &CudnnContext::Zero,
                        /* bias grad desc */ biasGradient_.getCudnnDescriptor(),
                        /* bias grad mem */ biasGradient_.deviceMemory->getMemoryHandle()));
                }

                // gradients of weights
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardFilter(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* outgrad desc */ outputGradient_.front().getCudnnDescriptor(),
                    /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv backward filter algo */ convolutionBackwardFilterAlgorithm_,
                    /* workspace mem */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* filter desc */ cudnnFilterDescriptor_,
                    /* weights gradient */ weightsGradient_.deviceMemory->getMemoryHandle()));

                // gradients of input data
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardData(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* filter desc */ cudnnFilterDescriptor_,
                    /* weights memory */ weights_.deviceMemory->getMemoryHandle(),
                    /* outgrad desc */ outputGradient_.front().getCudnnDescriptor(),
                    /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv backward data algo */ convolutionBackwardDataAlgorithm_,
                    /* workspace mem */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* inputgradient desc */ inputGradient_.front().getCudnnDescriptor(),
                    /* inputgradient mem */
                    inputGradient_.front().deviceMemory->getMemoryHandle()));
            }

            template class CudnnConvolution<float>;

            template <typename data_t>
            CudnnDeconvolution<data_t>::CudnnDeconvolution(
                const VolumeDescriptor& inputDescriptor, const VolumeDescriptor& outputDescriptor,
                const VolumeDescriptor& weightsDescriptor, const IndexVector_t& strides,
                const IndexVector_t& paddingLow, const IndexVector_t& paddingHigh, bool useBias,
                Initializer weightsInitializer, Initializer biasInitializer)
                : CudnnTrainable<data_t, false>(inputDescriptor, outputDescriptor,
                                                weightsDescriptor, useBias, weightsInitializer,
                                                biasInitializer)

            {
                // Create convolution descriptor
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreateConvolutionDescriptor(&convolutionDescriptor_));

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetConvolution2dDescriptor(
                    /* conv desc */ convolutionDescriptor_,
                    /* horizontal padding */ paddingLow[0],
                    /* vertical padding */ paddingHigh[0],
                    /* vertical stride */ strides[0],
                    /* horizontal stride */ strides[1],
                    /* dilation */ 1, 1,
                    /* conv mode */ CUDNN_CONVOLUTION,
                    /* type tag */ CUDNN_DATA_FLOAT));
            }

            template <typename data_t>
            CudnnDeconvolution<data_t>::~CudnnDeconvolution()
            {
                // Destroy conv descriptor
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnDestroyConvolutionDescriptor(convolutionDescriptor_));
            }

            template <typename data_t>
            void CudnnDeconvolution<data_t>::compileForwardStream()
            {
                BaseType::compileForwardStream();

                // Create filter descriptor for weights
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(
                    cudnnCreateFilterDescriptor(&cudnnFilterDescriptor_));

                // weights are in oihw format
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnSetFilter4dDescriptor(
                    cudnnFilterDescriptor_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                    weights_.getDimensions()[0], weights_.getDimensions()[1],
                    weights_.getDimensions()[2], weights_.getDimensions()[3]));

                // Since the workspace size we need to allocate may depend on
                // the forward and the backward pass, we choose the algorithm
                // for both here.

                std::size_t wsSize = 0;
                std::size_t tempSize = 0;
                // Choose forward algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionForwardAlgorithm(
                    cudnnContext_->getCudnnHandle(), output_.getCudnnDescriptor(),
                    cudnnFilterDescriptor_, convolutionDescriptor_,
                    input_.front().getCudnnDescriptor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
                    &convolutionForwardAlgorithm_));

                // Get forward workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionForwardWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), output_.getCudnnDescriptor(),
                    cudnnFilterDescriptor_, convolutionDescriptor_,
                    input_.front().getCudnnDescriptor(), convolutionForwardAlgorithm_, &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Choose backward data algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardDataAlgorithm(
                    cudnnContext_->getCudnnHandle(), cudnnFilterDescriptor_,
                    input_.front().getCudnnDescriptor(), convolutionDescriptor_,
                    output_.getCudnnDescriptor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
                    &convolutionBackwardDataAlgorithm_));

                // Get backward data workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), cudnnFilterDescriptor_,
                    input_.front().getCudnnDescriptor(), convolutionDescriptor_,
                    output_.getCudnnDescriptor(), convolutionBackwardDataAlgorithm_, &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Choose backward filter algorithm
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardFilterAlgorithm(
                    cudnnContext_->getCudnnHandle(), output_.getCudnnDescriptor(),
                    input_.front().getCudnnDescriptor(), convolutionDescriptor_,
                    cudnnFilterDescriptor_, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
                    &convolutionBackwardFilterAlgorithm_));

                // Get backward data workspace size
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    cudnnContext_->getCudnnHandle(), output_.getCudnnDescriptor(),
                    input_.front().getCudnnDescriptor(), convolutionDescriptor_,
                    cudnnFilterDescriptor_, convolutionBackwardFilterAlgorithm_, &tempSize));
                wsSize = std::max(wsSize, tempSize);

                // Allocate workspace memory
                workspaceMemory_ = std::make_shared<DeviceMemory<data_t>>(wsSize);
            }

            template <typename data_t>
            void CudnnDeconvolution<data_t>::forwardPropagate()
            {
                BaseType::validateForwardPropagation();

                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardData(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* weights desc */ cudnnFilterDescriptor_,
                    /* weights memory */ weights_.deviceMemory->getMemoryHandle(),
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv backward data algo */ convolutionBackwardDataAlgorithm_,
                    /* workspace mem */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* output desc */ output_.getCudnnDescriptor(),
                    /* output mem */ output_.deviceMemory->getMemoryHandle()));
                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnAddTensor(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* bias desc */ bias_.getCudnnDescriptor(),
                        /* bias memory */ bias_.deviceMemory->getMemoryHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* output desc */ output_.getCudnnDescriptor(),
                        /* output memory */
                        output_.deviceMemory->getMemoryHandle()));
                }
            }

            template <typename data_t>
            void CudnnDeconvolution<data_t>::backwardPropagate()
            {
                BaseType::validateBackwardPropagation();

                // Get gradient w.r.t. bias
                if (useBias_) {
                    ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardBias(
                        /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                        /* constant 1.f */ &CudnnContext::One,
                        /* outgrad desc */ outputGradient_.front().getCudnnDescriptor(),
                        /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                        /* constant 1.f */ &CudnnContext::Zero,
                        /* bias gradient descriptor */ biasGradient_.getCudnnDescriptor(),
                        /* bias gradient memory */ biasGradient_.deviceMemory->getMemoryHandle()));
                }

                // Get gradient w.r.t.weights
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionBackwardFilter(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* outgrad desc */ outputGradient_.front().getCudnnDescriptor(),
                    /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* input desc */ input_.front().getCudnnDescriptor(),
                    /* input memory */ input_.front().deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv backward filter algo */ convolutionBackwardFilterAlgorithm_,
                    /* workspace mem */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 1.f */ &CudnnContext::Zero,
                    /* weights desc */ cudnnFilterDescriptor_,
                    /* weights memory */ weightsGradient_.deviceMemory->getMemoryHandle()));

                // Get gradient w.r.t. input
                ELSA_ML_CHECK_CUDNN_BACKEND_STATUS(cudnnConvolutionForward(
                    /* cudnn handle */ cudnnContext_->getCudnnHandle(),
                    /* constant 1.f */ &CudnnContext::One,
                    /* outgrad desc */ outputGradient_.front().getCudnnDescriptor(),
                    /* outgrad mem */ outputGradient_.front().deviceMemory->getMemoryHandle(),
                    /* weights desc */ cudnnFilterDescriptor_,
                    /* weights memory */ weights_.deviceMemory->getMemoryHandle(),
                    /* conv desc */ convolutionDescriptor_,
                    /* conv fwd algo */ convolutionForwardAlgorithm_,
                    /* workspace mem */ workspaceMemory_->getMemoryHandle(),
                    /* workspace size */ workspaceMemory_->getSize(),
                    /* constant 0.f */ &CudnnContext::Zero,
                    /* ingrad desc */ inputGradient_.front().getCudnnDescriptor(),
                    /* ingrad mem */ inputGradient_.front().deviceMemory->getMemoryHandle()));
            }

            template class CudnnDeconvolution<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa