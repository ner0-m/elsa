#pragma once

#include "DnnlLayer.h"
namespace elsa
{
    template <typename data_t>
    class DnnlActivationLayer : public DnnlLayer<data_t>
    {
    public:
        void setAlpha(data_t alpha);
        void setBeta(data_t beta);

        virtual void compile() override;

    protected:
        DnnlActivationLayer(const DataDescriptor& inputDescriptor,
                            const DataDescriptor& outputDescriptor, dnnl::algorithm algorithm);

        // virtual void setInput(const DataContainer<data_t>& input) override;

        using BaseType = DnnlLayer<data_t>;

        using BaseType::_engine;
        using BaseType::_srcMemoryDescriptor;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_forwardPrimitives;

        dnnl::algorithm _algorithm;

        /// Primitive descriptor for element-wise forward propagation
        dnnl::eltwise_forward::primitive_desc _forwardPrimitiveDescriptor;

        data_t _alpha;
        data_t _beta;
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