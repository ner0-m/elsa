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

        DnnlTrainableLayer(const DataDescriptor& inputDescriptor,
                           const DataDescriptor& outputDescriptor,
                           const DataDescriptor& weightsDescriptor);

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

        /**
         * Set this layer's initializer that is used for initializing the layer's
         * weights and biases.
         *
         * \note This function must be used before DnnlTrainableLayer::compile
         * to have an effect.
         */
        void setInitializer(Initializer initializer);

    protected:
        using BaseType::_srcMemoryDescriptor;
        using BaseType::_reorderedSrcMemory;
        using BaseType::_dstMemoryDescriptor;
        using BaseType::_engine;
        using BaseType::_forwardPrimitives;
        using BaseType::_dstMemory;
        using BaseType::_srcMemory;
        using BaseType::_forwardArguments;
        using BaseType::_typeTag;
        using BaseType::_hasReorderedMemory;

        /// The dimension of the convolutional layer's weights
        dnnl::memory::dims _weightsDimensions;

        dnnl::memory::desc _weightsMemoryDescriptor;
        dnnl::memory _weightsMemory;
        dnnl::memory _reorderedWeightsMemory;

        dnnl::memory::dims _biasDimensions;
        dnnl::memory::desc _biasMemoryDescriptor;
        dnnl::memory _biasMemory;

        Initializer _initializer = Initializer::Uniform;
    };

} // namespace elsa