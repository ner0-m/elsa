#pragma once

#include "DnnlTrainableLayer.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "RandomInitializer.h"

#include "dnnl.hpp"

namespace elsa
{
    template <typename data_t>
    class DnnlDenseLayer final : public DnnlTrainableLayer<data_t>
    {
    public:
        /// \copydoc DnnlTrainableLayer::BaseType
        using BaseType = DnnlTrainableLayer<data_t>;

        DnnlDenseLayer(const DataDescriptor& inputDescriptor,
                       const DataDescriptor& outputDescriptor,
                       const DataDescriptor& weightsDescriptor, Initializer initializer);

    private:
        void compileForwardStream() override;

        void compileBackwardStream() override;

        /// \copydoc DnnlTrainableLayer::_weightsDimensions
        using BaseType::_weightsDimensions;

        /// \copydoc DnnlTrainableLayer::_weightsMemoryDescriptor
        using BaseType::_weightsMemoryDescriptor;

        /// \copydoc DnnlTrainableLayer::_weightsMemory
        using BaseType::_weightsMemory;

        /// \copydoc DnnlTrainableLayer::_reorderedWeightsMemory
        using BaseType::_reorderedWeightsMemory;

        /// \copydoc DnnlTrainableLayer::_biasDimensions
        using BaseType::_biasDimensions;

        /// \copydoc DnnlTrainableLayer::_biasMemoryDescriptor
        using BaseType::_biasMemoryDescriptor;

        /// \copydoc DnnlTrainableLayer::_biasMemory
        using BaseType::_biasMemory;

        /// \copydoc DnnlTrainableLayer::_srcMemoryDescriptor
        using BaseType::_srcMemoryDescriptor;

        /// \copydoc DnnlTrainableLayer::_reorderedSrcMemory
        using BaseType::_reorderedSrcMemory;

        /// \copydoc DnnlTrainableLayer::_dstMemoryDescriptor
        using BaseType::_dstMemoryDescriptor;

        /// \copydoc DnnlTrainableLayer::_engine
        using BaseType::_engine;

        /// \copydoc DnnlTrainableLayer::_forwardPrimitives
        using BaseType::_forwardPrimitives;

        /// \copydoc DnnlTrainableLayer::_dstMemory
        using BaseType::_dstMemory;

        /// \copydoc DnnlTrainableLayer::_srcMemory
        using BaseType::_srcMemory;

        /// \copydoc DnnlTrainableLayer::_forwardArguments
        using BaseType::_forwardArguments;

        /// \copydoc DnnlTrainableLayer::_typeTag
        using BaseType::_typeTag;

        /// \copydoc DnnlTrainableLayer::_hasReorderedMemory
        using BaseType::_hasReorderedMemory;

        using BaseType::_fanInOut;

        dnnl::inner_product_forward::primitive_desc _forwardPrimitiveDescriptor;
    };
} // namespace elsa