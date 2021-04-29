#pragma once

#include <vector>

#include "DnnlLayer.h"

namespace elsa::ml
{
    namespace detail
    {
        template <typename data_t>
        class DnnlMerging : public DnnlLayer<data_t>
        {
        public:
            DnnlMerging(const std::vector<VolumeDescriptor>& inputDescriptors,
                        const VolumeDescriptor& outputDescriptors);

            bool needsForwardSynchronisation() const override;

            bool canMerge() const override;

        protected:
            using BaseType = DnnlLayer<data_t>;

            /// \copydoc DnnlLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlLayer::_inputDescriptor
            using BaseType::_inputDescriptor;

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
        };

        template <typename data_t>
        class DnnlSum : public DnnlMerging<data_t>
        {
        public:
            DnnlSum(const std::vector<VolumeDescriptor>& inputDescriptors,
                    const VolumeDescriptor& outputDescriptors);

            /// Execute this layer's backward primitives on executionStream
            void backwardPropagate(dnnl::stream& executionStream) override;

        private:
            /// \copydoc DnnlLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlLayer::compileBackwardStream
            void compileBackwardStream() override;

            using BaseType = DnnlMerging<data_t>;

            /// \copydoc DnnlLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlLayer::_inputDescriptor
            using BaseType::_inputDescriptor;

            /// \copydoc DnnlLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlLayer::_outputDescriptor
            using BaseType::_outputDescriptor;

            /// \copydoc DnnlLayer::_outputGradient
            using BaseType::_outputGradient;

            /// \copydoc DnnlLayer::_forwardStream
            using BaseType::_forwardStream;

            /// \copydoc DnnlLayer::_backwardStream
            using BaseType::_backwardStream;

            /// \copydoc DnnlLayer::_engine
            using BaseType::_engine;

            dnnl::sum::primitive_desc _forwardPrimitiveDescriptor;
        };

        template <typename data_t>
        class DnnlConcatenate : public DnnlMerging<data_t>
        {
        public:
            DnnlConcatenate(index_t axis, const std::vector<VolumeDescriptor>& inputDescriptors,
                            const VolumeDescriptor& outputDescriptors);

            /// Execute this layer's backward primitives on executionStream
            void backwardPropagate(dnnl::stream& executionStream) override;

        private:
            /// \copydoc DnnlLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlLayer::compileBackwardStream
            void compileBackwardStream() override;

            using BaseType = DnnlMerging<data_t>;

            /// \copydoc DnnlLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlLayer::_inputDescriptor
            using BaseType::_inputDescriptor;

            /// \copydoc DnnlLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlLayer::_outputDescriptor
            using BaseType::_outputDescriptor;

            /// \copydoc DnnlLayer::_outputGradient
            using BaseType::_outputGradient;

            /// \copydoc DnnlLayer::_forwardStream
            using BaseType::_forwardStream;

            /// \copydoc DnnlLayer::_backwardStream
            using BaseType::_backwardStream;

            /// \copydoc DnnlLayer::_engine
            using BaseType::_engine;

            dnnl::concat::primitive_desc _forwardPrimitiveDescriptor;

            index_t _axis;
        };
    } // namespace detail
} // namespace elsa::ml