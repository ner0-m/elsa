#pragma once

#include "DnnlLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        class DnnlSoftmaxLayer : public DnnlLayer<data_t>
        {
        public:
            DnnlSoftmaxLayer(const VolumeDescriptor& inputDescriptor,
                             const VolumeDescriptor& outputDescriptor);

        private:
            /// DnnlLayer::compileForwardStream
            void compileForwardStream() override;

            /// DnnlLayer::compileBackwardStream
            void compileBackwardStream() override;

            using BaseType = DnnlLayer<data_t>;

            /// DnnlLayer::_input
            using BaseType::_input;

            /// DnnlLayer::_inputGradient
            using BaseType::_inputGradient;

            /// DnnlLayer::_output
            using BaseType::_output;

            /// DnnlLayer::_outputGradient
            using BaseType::_outputGradient;

            /// DnnlLayer::_forwardStream
            using BaseType::_forwardStream;

            /// DnnlLayer::_backwardStream
            using BaseType::_backwardStream;

            /// DnnlLayer::_engine
            using BaseType::_engine;

            /// The input axis along that this layer calculates softmax
            int _softmaxAxis;

            /// This layer's forward primitive descriptor
            dnnl::softmax_forward::primitive_desc _forwardPrimitiveDescriptor;

            /// This layer's backward primitive descriptor
            dnnl::softmax_backward::primitive_desc _backwardPrimitiveDescriptor;
        };
    } // namespace detail
} // namespace elsa::ml