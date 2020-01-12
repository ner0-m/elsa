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
        void compileForwardStream() override;

        void compileBackwardStream() override;
        void compileBackwardDataStream();
        void compileBackwardWeightsStream();

        using BaseType::_typeTag;

        using BaseType::_engine;

        using BaseType::_input;
        using BaseType::_inputGradient;

        using BaseType::_output;
        using BaseType::_outputGradient;

        using BaseType::_forwardStream;
        using BaseType::_backwardStream;

        using BaseType::_weights;
        using BaseType::_weightsGradient;

        using BaseType::_bias;
        using BaseType::_biasGradient;

        dnnl::memory::dims _paddingDimensions;

        dnnl::memory::dims _strideDimensions;

        dnnl::convolution_forward::primitive_desc _forwardPrimitiveDescriptor;
        dnnl::convolution_backward_weights::primitive_desc _backwardWeightsPrimitiveDescriptor;
        dnnl::convolution_backward_data::primitive_desc _backwardDataPrimitiveDescriptor;
    };
} // namespace elsa