#pragma once

#include "DnnlLayer.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "RandomInitializer.h"

#include "dnnl.hpp"

namespace elsa
{
    /**
     * Trainable Dnnl layer.
     *
     * This layer is used as a base class for all Dnnl layer's with trainable
     * parameters, such as convolutional or dense layers.
     *
     * \tparam data_t Type of all coefficiens used in the layer
     */
    template <typename data_t>
    class DnnlTrainableLayer : public DnnlLayer<data_t>
    {
    public:
        /// Type of this layer's base class
        using BaseType = DnnlLayer<data_t>;

        /**
         * Construct a trainable Dnnl network layer by passing a descriptor for its input, its
         * output and weights and an initializer for its weights and biases.
         */
        DnnlTrainableLayer(const DataDescriptor& inputDescriptor,
                           const DataDescriptor& outputDescriptor,
                           const DataDescriptor& weightsDescriptor, Initializer initializer);

        /**
         * Set this layer's weights by passing a DataContainer.
         *
         * \note This functions performs a copy from a DataContainer to Dnnl
         * memory and should therefore be used for debugging or testing purposes
         * only. The layer is capable of initializing its weights on its own.
         */
        void setWeights(const DataContainer<data_t>& weights);

        /**
         * Set this layer's biases by passing a DataContainer.
         *
         * \note This functions performs a copy from a DataContainer to Dnnl
         * memory and should therefore be used for debugging or testing purposes
         * only. The layer is capable of initializing its biases on its own.
         */
        void setBias(const DataContainer<data_t>& bias);

        DataContainer<data_t> getGradientWeights() const;
        DataContainer<data_t> getGradientBias() const;

    protected:
        void compileForwardStream() override;
        void compileBackwardStream() override;

        /// \copydoc DnnlLayer::_srcMemoryDescriptor
        using BaseType::_srcMemoryDescriptor;

        /// \copydoc DnnlLayer::_reorderedSrcMemory
        using BaseType::_reorderedSrcMemory;

        /// \copydoc DnnlLayer::_dstMemoryDescriptor
        using BaseType::_dstMemoryDescriptor;

        /// \copydoc DnnlLayer::_engine
        using BaseType::_engine;

        /// \copydoc DnnlLayer::_forwardPrimitives
        using BaseType::_forwardPrimitives;

        /// \copydoc DnnlLayer::_dstMemory
        using BaseType::_dstMemory;

        /// \copydoc DnnlLayer::_srcMemory
        using BaseType::_srcMemory;

        /// \copydoc DnnlLayer::_forwardArguments
        using BaseType::_forwardArguments;

        /// \copydoc DnnlLayer::_typeTag
        using BaseType::_typeTag;

        /// \copydoc DnnlLayer::_hasReorderedMemory
        using BaseType::_hasReorderedMemory;

        using BaseType::_gradientSrcMemoryDescriptor;
        using BaseType::_gradientDstMemoryDescriptor;
        using BaseType::_reorderedGradientDstMemory;
        using BaseType::_gradientDstMemory;
        using BaseType::_backwardPrimitives;
        using BaseType::_backwardArguments;
        using BaseType::_gradientSrcMemory;

        std::unique_ptr<DataDescriptor> _weightsDescriptor;
        std::unique_ptr<DataDescriptor> _biasDescriptor;

        /// The dimension of the convolutional layer's weights
        dnnl::memory::dims _weightsDimensions;

        /// This layer's weights memory descriptor
        dnnl::memory::desc _weightsMemoryDescriptor;

        /// This layer's weights memory
        dnnl::memory _weightsMemory;

        /// This layer's weights memory after possible reordering
        dnnl::memory _reorderedWeightsMemory;

        dnnl::memory::format_tag _weightsMemoryFormatTag;

        dnnl::memory::dims _biasDimensions;
        dnnl::memory::desc _biasMemoryDescriptor;
        dnnl::memory _biasMemory;

        Initializer _initializer;
        typename RandomInitializer<data_t>::FanPairType _fanInOut;

        dnnl::memory _gradientWeightsMemory;
        dnnl::memory _reorderedGradientWeightsMemory;
        dnnl::memory::desc _gradientWeightsMemoryDescriptor;

        dnnl::memory _gradientBiasMemory;
        dnnl::memory::desc _gradientBiasMemoryDescriptor;
    };

} // namespace elsa