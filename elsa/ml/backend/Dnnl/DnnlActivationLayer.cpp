#include "DnnlActivationLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        DnnlActivationLayer<data_t>::DnnlActivationLayer(const VolumeDescriptor& inputDescriptor,
                                                         const VolumeDescriptor& outputDescriptor,
                                                         dnnl::algorithm algorithm)
            : DnnlLayer<data_t>(inputDescriptor, outputDescriptor, "DnnlActivationLayer"),
              algorithm_(algorithm)
        {
        }

        template <typename data_t>
        void DnnlActivationLayer<data_t>::setAlpha(data_t alpha)
        {
            _alpha = alpha;
        }

        template <typename data_t>
        void DnnlActivationLayer<data_t>::setBeta(data_t beta)
        {
            _beta = beta;
        }

        template <typename data_t>
        void DnnlActivationLayer<data_t>::compileForwardStream()
        {
            BaseType::compileForwardStream();

            // Set forward primitive description
            auto desc = dnnl::eltwise_forward::desc(
                /* Inference type */ dnnl::prop_kind::forward_training,
                /* Element-wise algorithm */ algorithm_,
                /* Source memory descriptor */ _input.front().descriptor,
                /* Alpha parameter */ _alpha,
                /* Beta parameter */ _beta);

            _forwardPrimitiveDescriptor = dnnl::eltwise_forward::primitive_desc(desc, *_engine);

            // Set forward primitive
            ELSA_ML_ADD_DNNL_PRIMITIVE(_forwardStream,
                                       dnnl::eltwise_forward(_forwardPrimitiveDescriptor));

            // Set output memory. Since no activation layer can reorder we set effective memory
            // directly
            _output.effectiveMemory =
                std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

            _forwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.front().effectiveMemory},
                                                {DNNL_ARG_DST, *_output.effectiveMemory}});
        }

        template <typename data_t>
        void DnnlActivationLayer<data_t>::compileBackwardStream()
        {
            BaseType::compileBackwardStream();

            auto desc = dnnl::eltwise_backward::desc(
                /* Element-wise algorithm */ algorithm_,
                /* Gradient dst memory descriptor */ _outputGradient.front().descriptor,
                /* Source memory descriptor */ _input.front().descriptor,
                /* Alpha parameter */ _alpha,
                /* Beta parameter */ _beta);

            _backwardPrimitiveDescriptor =
                dnnl::eltwise_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

            // Reorder if necessary
            this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(),
                                _outputGradient.front(), _backwardStream);

            _inputGradient.front().effectiveMemory = std::make_shared<dnnl::memory>(
                _backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

            _outputGradient.front().effectiveMemory = _outputGradient.front().describedMemory;
            BaseType::validateDnnlMemory(_input.front().effectiveMemory);
            BaseType::validateDnnlMemory(_outputGradient.front().effectiveMemory);
            BaseType::validateDnnlMemory(_outputGradient.front().describedMemory);
            BaseType::validateDnnlMemory(_inputGradient.front().effectiveMemory);

            ELSA_ML_ADD_DNNL_PRIMITIVE(_backwardStream,
                                       dnnl::eltwise_backward(_backwardPrimitiveDescriptor));
            _backwardStream.arguments.push_back(
                {/* Input */
                 {DNNL_ARG_SRC, *_input.front().effectiveMemory},
                 {DNNL_ARG_DIFF_DST, *_outputGradient.front().effectiveMemory},
                 /* Output */
                 {DNNL_ARG_DIFF_SRC, *_inputGradient.front().effectiveMemory}});
        }

        template <typename data_t>
        DnnlAbs<data_t>::DnnlAbs(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_abs)
        {
        }

        template <typename data_t>
        DnnlBoundedRelu<data_t>::DnnlBoundedRelu(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_bounded_relu)
        {
        }

        template <typename data_t>
        DnnlElu<data_t>::DnnlElu(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_elu)
        {
        }

        template <typename data_t>
        DnnlExp<data_t>::DnnlExp(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_exp)
        {
        }

        template <typename data_t>
        DnnlGelu<data_t>::DnnlGelu(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_gelu)
        {
        }

        template <typename data_t>
        DnnlLinear<data_t>::DnnlLinear(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_linear)
        {
        }

        template <typename data_t>
        DnnlLogistic<data_t>::DnnlLogistic(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_logistic)
        {
        }

        template <typename data_t>
        DnnlRelu<data_t>::DnnlRelu(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_relu)
        {
        }

        template <typename data_t>
        DnnlSoftRelu<data_t>::DnnlSoftRelu(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_soft_relu)
        {
        }

        template <typename data_t>
        DnnlSqrt<data_t>::DnnlSqrt(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_sqrt)
        {
        }

        template <typename data_t>
        DnnlSquare<data_t>::DnnlSquare(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_square)
        {
        }

        template <typename data_t>
        DnnlSwish<data_t>::DnnlSwish(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_swish)
        {
        }

        template <typename data_t>
        DnnlTanh<data_t>::DnnlTanh(const VolumeDescriptor& inputDescriptor)
            : DnnlActivationLayer<data_t>(inputDescriptor, inputDescriptor,
                                          dnnl::algorithm::eltwise_tanh)
        {
        }

        template class DnnlActivationLayer<float>;

        template struct DnnlAbs<float>;
        template struct DnnlBoundedRelu<float>;
        template struct DnnlElu<float>;
        template struct DnnlExp<float>;
        template struct DnnlLinear<float>;
        template struct DnnlGelu<float>;
        template struct DnnlLogistic<float>;
        template struct DnnlRelu<float>;
        template struct DnnlSoftRelu<float>;
        template struct DnnlSqrt<float>;
        template struct DnnlSquare<float>;
        template struct DnnlSwish<float>;
        template struct DnnlTanh<float>;

    } // namespace detail
} // namespace elsa::ml
