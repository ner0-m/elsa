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

        /// Get this layer's weights gradient as a DataContainer.
        DataContainer<data_t> getGradientWeights() const;

        /// Get this layer's bias gradient as a DataContainer.
        DataContainer<data_t> getGradientBias() const;

    protected:
        /// \copydoc DnnlLayer::compileForwardStream
        void compileForwardStream() override;

        /// \copydoc DnnlLayer::compileBackwardStream
        void compileBackwardStream() override;

        using DnnlMemory = typename BaseType::DnnlMemory;

        /// \copydoc DnnlLayer::_typeTag
        using BaseType::_typeTag;

        /// \copydoc DnnlLayer::_engine
        using BaseType::_engine;

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

        /// This layer's weights memory
        DnnlMemory _weights;

        /// This layer's weights gradient memory
        DnnlMemory _weightsGradient;

        /// This layer's bias memory
        DnnlMemory _bias;

        /// This layer's bias gradient memory
        DnnlMemory _biasGradient;

        /// This layer's weights DataDescriptor
        std::unique_ptr<DataDescriptor> _weightsDescriptor;

        /// This layer's bias DataDescriptor
        std::unique_ptr<DataDescriptor> _biasDescriptor;

        /// This layer's initializer tag
        Initializer _initializer;

        /// This layer's fanIn/fanOut pair that is used during random initialization of weights and
        /// biases
        typename RandomInitializer<data_t>::FanPairType _fanInOut;
    };

} // namespace elsa