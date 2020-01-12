#pragma once

#include "DnnlTrainableLayer.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "RandomInitializer.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlConvLayer final : public DnnlTrainableLayer<data_t>
    {
    public:
        using BaseType = DnnlTrainableLayer<data_t>;

        DnnlConvLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& outputDescriptor,
                      const DataDescriptor& weightsDescriptor, const IndexVector_t& strideVector,
                      const IndexVector_t& paddingVector, Initializer initializer);

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

        dnnl::memory::dims _paddingDimensions;

        dnnl::memory::dims _strideDimensions;

        dnnl::convolution_forward::primitive_desc _forwardPrimitiveDescriptor;
        dnnl::convolution_backward_weights::primitive_desc _backwardWeightsPrimitiveDescriptor;
        dnnl::convolution_backward_data::primitive_desc _backwardDataPrimitiveDescriptor;
    };
} // namespace elsa