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

        using DnnlMemory = typename BaseType::DnnlMemory;

        using BaseType::_typeTag;

        using BaseType::_engine;

        using BaseType::_input;
        using BaseType::_inputGradient;

        using BaseType::_output;
        using BaseType::_outputGradient;

        using BaseType::_forwardStream;
        using BaseType::_backwardStream;

        DnnlMemory _weights;
        DnnlMemory _weightsGradient;

        DnnlMemory _bias;
        DnnlMemory _biasGradient;

        std::unique_ptr<DataDescriptor> _weightsDescriptor;
        std::unique_ptr<DataDescriptor> _biasDescriptor;

        Initializer _initializer;
        typename RandomInitializer<data_t>::FanPairType _fanInOut;
    };

} // namespace elsa