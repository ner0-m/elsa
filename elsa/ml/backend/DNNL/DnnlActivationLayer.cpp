#include "DnnlActivationLayer.h"

namespace elsa
{
    template <typename data_t>
    DnnlActivationLayer<data_t>::DnnlActivationLayer(const DataDescriptor& inputDescriptor,
                                                     const DataDescriptor& outputDescriptor,
                                                     dnnl::algorithm algorithm)
        : DnnlLayer<data_t>(inputDescriptor, outputDescriptor), _algorithm(algorithm)
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

    static void validateDnnlMemory(std::shared_ptr<dnnl::memory> mem)
    {
        assert(mem && "Pointer to memory cannot be null");
        assert(mem->get_desc().get_size() && "Memory cannot have size 0,");
    }

    template <typename data_t>
    void DnnlActivationLayer<data_t>::compileBackwardStream()
    {
        auto desc = dnnl::eltwise_backward::desc(
            /* Element-wise algorithm */ _algorithm,
            /* Gradient dst memory descriptor */ _outputGradient.descriptor,
            /* Source memory descriptor */ _input.descriptor,
            /* Alpha parameter */ _alpha,
            /* Beta parameter */ _beta);

        _backwardPrimitiveDescriptor =
            dnnl::eltwise_backward::primitive_desc(desc, *_engine, _forwardPrimitiveDescriptor);

        // Reorder if necessary
        this->reorderMemory(_backwardPrimitiveDescriptor.diff_dst_desc(), _outputGradient,
                            _backwardStream);

        _inputGradient.effectiveMemory =
            std::make_shared<dnnl::memory>(_backwardPrimitiveDescriptor.diff_src_desc(), *_engine);

        validateDnnlMemory(_input.effectiveMemory);
        validateDnnlMemory(_outputGradient.effectiveMemory);
        validateDnnlMemory(_inputGradient.effectiveMemory);

        _backwardStream.primitives.push_back(dnnl::eltwise_backward(_backwardPrimitiveDescriptor));
        _backwardStream.arguments.push_back({{DNNL_ARG_SRC, *_input.effectiveMemory},
                                             {DNNL_ARG_DIFF_DST, *_outputGradient.effectiveMemory},
                                             {DNNL_ARG_DIFF_SRC, *_inputGradient.effectiveMemory}});
    }

    template <typename data_t>
    void DnnlActivationLayer<data_t>::compileForwardStream()
    {
        // Set forward primitive description
        auto desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_training, _algorithm,
                                                _input.descriptor, _alpha, _beta);

        _forwardPrimitiveDescriptor = dnnl::eltwise_forward::primitive_desc(desc, *_engine);

        // Set forward primitive
        _forwardStream.primitives.push_back(dnnl::eltwise_forward(_forwardPrimitiveDescriptor));

        // Set output memory. Since no activation layer can reorder we set effective memory directly
        _output.effectiveMemory =
            std::make_shared<dnnl::memory>(_forwardPrimitiveDescriptor.dst_desc(), *_engine);

        _forwardStream.arguments.push_back(
            {{DNNL_ARG_SRC, *_input.effectiveMemory}, {DNNL_ARG_DST, *_output.effectiveMemory}});
    }

    template <typename data_t>
    DnnlAbs<data_t>::DnnlAbs(const DataDescriptor& inputDescriptor,
                             const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_abs)
    {
    }

    template <typename data_t>
    DnnlBoundedRelu<data_t>::DnnlBoundedRelu(const DataDescriptor& inputDescriptor,
                                             const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_bounded_relu)
    {
    }

    template <typename data_t>
    DnnlElu<data_t>::DnnlElu(const DataDescriptor& inputDescriptor,
                             const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_elu)
    {
    }

    template <typename data_t>
    DnnlExp<data_t>::DnnlExp(const DataDescriptor& inputDescriptor,
                             const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_exp)
    {
    }

    template <typename data_t>
    DnnlGelu<data_t>::DnnlGelu(const DataDescriptor& inputDescriptor,
                               const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_gelu)
    {
    }

    template <typename data_t>
    DnnlLinear<data_t>::DnnlLinear(const DataDescriptor& inputDescriptor,
                                   const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_linear)
    {
    }

    template <typename data_t>
    DnnlLogistic<data_t>::DnnlLogistic(const DataDescriptor& inputDescriptor,
                                       const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_logistic)
    {
    }

    template <typename data_t>
    DnnlRelu<data_t>::DnnlRelu(const DataDescriptor& inputDescriptor,
                               const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_relu)
    {
    }

    template <typename data_t>
    DnnlSoftRelu<data_t>::DnnlSoftRelu(const DataDescriptor& inputDescriptor,
                                       const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_soft_relu)
    {
    }

    template <typename data_t>
    DnnlSqrt<data_t>::DnnlSqrt(const DataDescriptor& inputDescriptor,
                               const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_sqrt)
    {
    }

    template <typename data_t>
    DnnlSquare<data_t>::DnnlSquare(const DataDescriptor& inputDescriptor,
                                   const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_square)
    {
    }

    template <typename data_t>
    DnnlSwish<data_t>::DnnlSwish(const DataDescriptor& inputDescriptor,
                                 const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
                                      dnnl::algorithm::eltwise_swish)
    {
    }

    template <typename data_t>
    DnnlTanh<data_t>::DnnlTanh(const DataDescriptor& inputDescriptor,
                               const DataDescriptor& outputDescriptor)
        : DnnlActivationLayer<data_t>(inputDescriptor, outputDescriptor,
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

} // namespace elsa
