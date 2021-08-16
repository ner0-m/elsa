#pragma once

#include "DnnlLayer.h"
#include "VolumeDescriptor.h"

namespace elsa::ml
{
    namespace detail
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
            DnnlActivationLayer(const VolumeDescriptor& inputDescriptor,
                                const VolumeDescriptor& outputDescriptor,
                                dnnl::algorithm algorithm);

            /// \copydoc DnnlLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlLayer::compileBackwardStream
            void compileBackwardStream() override;

            using BaseType = DnnlLayer<data_t>;

            /// \copydoc DnnlLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlLayer::_outputGradient
            using BaseType::_outputGradient;

            /// \copydoc DnnlLayer::_forwardStream
            using BaseType::_forwardStream;

            /// \copydoc DnnlLayer::_backwardStream
            using BaseType::_backwardStream;

            /// \copydoc DnnlLayer::_engine
            using BaseType::_engine;

            dnnl::algorithm algorithm_;

            /// Primitive descriptor for element-wise forward propagation
            dnnl::eltwise_forward::primitive_desc _forwardPrimitiveDescriptor;

            /// Primitive descriptor for element-wise backward propagation
            dnnl::eltwise_backward::primitive_desc _backwardPrimitiveDescriptor;

            data_t _alpha = static_cast<data_t>(0);
            data_t _beta = static_cast<data_t>(0);
        };

        template <typename data_t>
        struct DnnlAbs final : public DnnlActivationLayer<data_t> {
            explicit DnnlAbs(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlBoundedRelu final : public DnnlActivationLayer<data_t> {
            explicit DnnlBoundedRelu(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlElu final : public DnnlActivationLayer<data_t> {
            explicit DnnlElu(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlExp final : public DnnlActivationLayer<data_t> {
            explicit DnnlExp(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlGelu final : public DnnlActivationLayer<data_t> {
            explicit DnnlGelu(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlLinear final : public DnnlActivationLayer<data_t> {
            explicit DnnlLinear(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlLogistic final : public DnnlActivationLayer<data_t> {
            explicit DnnlLogistic(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlRelu final : public DnnlActivationLayer<data_t> {
            explicit DnnlRelu(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlSoftRelu final : public DnnlActivationLayer<data_t> {
            explicit DnnlSoftRelu(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlSqrt final : public DnnlActivationLayer<data_t> {
            explicit DnnlSqrt(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlSquare final : public DnnlActivationLayer<data_t> {
            explicit DnnlSquare(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlSwish final : public DnnlActivationLayer<data_t> {
            explicit DnnlSwish(const VolumeDescriptor& inputDescriptor);
        };

        template <typename data_t>
        struct DnnlTanh final : public DnnlActivationLayer<data_t> {
            explicit DnnlTanh(const VolumeDescriptor& inputDescriptor);
        };
    } // namespace detail
} // namespace elsa::ml
