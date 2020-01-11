#pragma once

#include "DnnlLayer.h"
#include "DataDescriptor.h"

namespace elsa
{
    template <typename data_t>
    class DnnlActivationLayer : public DnnlLayer<data_t>
    {
    public:
        /// Set the layer's alpha parameter.
        void setAlpha(data_t alpha);

        /// Set the layer's beta parameter.
        void setBeta(data_t beta);

    protected:
        DnnlActivationLayer(const DataDescriptor& inputDescriptor,
                            const DataDescriptor& outputDescriptor, dnnl::algorithm algorithm);

        void compileBackwardStream() override;
        void compileForwardStream() override;

        using BaseType = DnnlLayer<data_t>;

        using BaseType::_input;
        using BaseType::_inputGradient;

        using BaseType::_output;
        using BaseType::_outputGradient;

        using BaseType::_forwardStream;
        using BaseType::_backwardStream;

        using BaseType::_engine;

        dnnl::algorithm _algorithm;

        /// Primitive descriptor for element-wise forward propagation
        dnnl::eltwise_forward::primitive_desc _forwardPrimitiveDescriptor;

        /// Primitive descriptor for element-wise backward propagation
        dnnl::eltwise_backward::primitive_desc _backwardPrimitiveDescriptor;

        data_t _alpha = static_cast<data_t>(0);
        data_t _beta = static_cast<data_t>(0);
    };

    template <typename data_t>
    struct DnnlAbs final : public DnnlActivationLayer<data_t> {
        DnnlAbs(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlBoundedRelu final : public DnnlActivationLayer<data_t> {
        DnnlBoundedRelu(const DataDescriptor& inputDescriptor,
                        const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlElu final : public DnnlActivationLayer<data_t> {
        DnnlElu(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlExp final : public DnnlActivationLayer<data_t> {
        DnnlExp(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlGelu final : public DnnlActivationLayer<data_t> {
        DnnlGelu(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlLinear final : public DnnlActivationLayer<data_t> {
        DnnlLinear(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlLogistic final : public DnnlActivationLayer<data_t> {
        DnnlLogistic(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlRelu final : public DnnlActivationLayer<data_t> {
        DnnlRelu(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlSoftRelu final : public DnnlActivationLayer<data_t> {
        DnnlSoftRelu(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlSqrt final : public DnnlActivationLayer<data_t> {
        DnnlSqrt(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlSquare final : public DnnlActivationLayer<data_t> {
        DnnlSquare(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlSwish final : public DnnlActivationLayer<data_t> {
        DnnlSwish(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };

    template <typename data_t>
    struct DnnlTanh final : public DnnlActivationLayer<data_t> {
        DnnlTanh(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor);
    };
} // namespace elsa
