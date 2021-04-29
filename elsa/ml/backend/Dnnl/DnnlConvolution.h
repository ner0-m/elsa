#pragma once

#include "DnnlTrainableLayer.h"
#include "DataDescriptor.h"
#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "Initializer.h"

#include "dnnl.hpp"

namespace elsa::ml
{
    namespace detail
    {
        /// A Convolutional Dnnl backend layer
        template <typename data_t>
        class DnnlConvolution final : public DnnlTrainableLayer<data_t>
        {
        public:
            using BaseType = DnnlTrainableLayer<data_t>;

            /// Construct a Dnnl Convolutional layer
            ///
            /// \param inputDescriptor Descriptor for the layer's input
            /// \param outputDescriptor Descriptor for the layer's output
            /// \param weightsDescriptor Descriptor for the layer's weights
            /// \param strides Vector containing strides to perform a convolution operation with
            /// \param paddingLow Vector containing the input padding for lower spatial dimensions
            /// \param paddingHigh Vector containing the input padding for higher spatial dimensions
            DnnlConvolution(const VolumeDescriptor& inputDescriptor,
                            const VolumeDescriptor& outputDescriptor,
                            const VolumeDescriptor& weightsDescriptor, const IndexVector_t& strides,
                            const IndexVector_t& paddingLow, const IndexVector_t& paddingHigh,
                            Initializer initializer);

        private:
            /// \copydoc DnnlTrainableLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlTrainableLayer::compileBackwardStream
            void compileBackwardStream() override;

            /// Compile this layer's data backward stream
            void compileBackwardDataStream();

            /// Compile this layer's weights backward stream
            void compileBackwardWeightsStream();

            /// \copydoc DnnlTrainableLayer::_typeTag
            using BaseType::_typeTag;

            /// \copydoc DnnlTrainableLayer::_engine
            using BaseType::_engine;

            /// \copydoc DnnlTrainableLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlTrainableLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlTrainableLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlTrainableLayer::_outputGradient
            using BaseType::_outputGradient;

            /// \copydoc DnnlTrainableLayer::_forwardStream
            using BaseType::_forwardStream;

            /// \copydoc DnnlTrainableLayer::_backwardStream
            using BaseType::_backwardStream;

            /// \copydoc DnnlTrainableLayer::_weights
            using BaseType::_weights;

            /// \copydoc DnnlTrainableLayer::_weightsGradient
            using BaseType::_weightsGradient;

            /// \copydoc DnnlTrainableLayer::_bias
            using BaseType::_bias;

            /// \copydoc DnnlTrainableLayer::_biasGradient
            using BaseType::_biasGradient;

            /// Padding for lower spatial dimensions
            dnnl::memory::dims _paddingLowDimensions;

            /// Padding for higher spatial dimensions
            dnnl::memory::dims _paddingHighDimensions;

            /// Convolution strides
            dnnl::memory::dims _stridesDimensions;

            /// Descriptor for primitive performing forward propagation
            dnnl::convolution_forward::primitive_desc _forwardPrimitiveDescriptor;

            /// Descriptor for primitive performing backward propagation of weights
            dnnl::convolution_backward_weights::primitive_desc _backwardWeightsPrimitiveDescriptor;

            /// Descriptor for primitive performing backward propagation of data
            dnnl::convolution_backward_data::primitive_desc _backwardDataPrimitiveDescriptor;
        };

        template <typename data_t>
        class DnnlDeconvolution final : public DnnlTrainableLayer<data_t>
        {
        public:
            using BaseType = DnnlTrainableLayer<data_t>;

            DnnlDeconvolution(const VolumeDescriptor& inputDescriptor,
                              const VolumeDescriptor& outputDescriptor,
                              const VolumeDescriptor& weightsDescriptor,
                              const IndexVector_t& strides, const IndexVector_t& paddingLow,
                              const IndexVector_t& paddingHigh, Initializer initializer);

        private:
            /// \copydoc DnnlTrainableLayer::compileForwardStream
            void compileForwardStream() override;

            /// \copydoc DnnlTrainableLayer::compileBackwardStream
            void compileBackwardStream() override;

            /// Compile this layer's data backward stream
            void compileBackwardDataStream();

            /// Compile this layer's weights backward stream
            void compileBackwardWeightsStream();

            /// \copydoc DnnlTrainableLayer::_typeTag
            using BaseType::_typeTag;

            /// \copydoc DnnlTrainableLayer::_engine
            using BaseType::_engine;

            /// \copydoc DnnlTrainableLayer::_input
            using BaseType::_input;

            /// \copydoc DnnlTrainableLayer::_inputGradient
            using BaseType::_inputGradient;

            /// \copydoc DnnlTrainableLayer::_output
            using BaseType::_output;

            /// \copydoc DnnlTrainableLayer::_outputGradient
            using BaseType::_outputGradient;

            /// \copydoc DnnlTrainableLayer::_forwardStream
            using BaseType::_forwardStream;

            /// \copydoc DnnlTrainableLayer::_backwardStream
            using BaseType::_backwardStream;

            /// \copydoc DnnlTrainableLayer::_weights
            using BaseType::_weights;

            /// \copydoc DnnlTrainableLayer::_weightsGradient
            using BaseType::_weightsGradient;

            /// \copydoc DnnlTrainableLayer::_bias
            using BaseType::_bias;

            /// \copydoc DnnlTrainableLayer::_biasGradient
            using BaseType::_biasGradient;

            /// Padding for lower spatial dimensions
            dnnl::memory::dims _paddingLowDimensions;

            /// Padding for higher spatial dimensions
            dnnl::memory::dims _paddingHighDimensions;

            /// Convolution strides
            dnnl::memory::dims _stridesDimensions;

            /// Descriptor for primitive performing forward propagation
            dnnl::deconvolution_forward::primitive_desc _forwardPrimitiveDescriptor;

            /// Descriptor for primitive performing backward propagation of weights
            dnnl::deconvolution_backward_weights::primitive_desc
                _backwardWeightsPrimitiveDescriptor;

            /// Descriptor for primitive performing backward propagation of data
            dnnl::deconvolution_backward_data::primitive_desc _backwardDataPrimitiveDescriptor;
        };
    } // namespace detail
} // namespace elsa::ml