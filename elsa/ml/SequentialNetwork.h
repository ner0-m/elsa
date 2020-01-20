#pragma once

#include <vector>
#include <memory>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "Layer.h"
#include "ActivationLayer.h"
#include "PoolingLayer.h"
#include "ConvLayer.h"
#include "DenseLayer.h"
#include "FixedLayer.h"
#include "SoftmaxLayer.h"
#include "LRNLayer.h"
#include "RandomInitializer.h"
#include "DnnlSequentialNetwork.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class SequentialNetwork final
    {
    public:
        using BackendNetworkType = typename detail::BackendSelector<SequentialNetwork>::Type;

        /**
         * Construct a sequential network from a given DataDescriptor that describes the network
         * input.
         *
         * \param[in] inputDescriptor DataDescriptor for the network's input
         *
         * \note This function throws if the inputDescriptor has a dimension > 5
         */
        SequentialNetwork(const DataDescriptor& inputDescriptor);

        /**
         * Add a dense layer to the network.
         *
         * \param[in] numNeurons Number of neurons of the dense layer
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>&
            addDenseLayer(int numNeurons, Initializer initializer = Initializer::Uniform);

        /**
         * Add a pooling layer to the network.
         *
         * \param[in] poolingWindow Pooling window of the pooling layer
         * \param[in] poonlingStide Pooling stride of the pooling layer+
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>& addPoolingLayer(const IndexVector_t& poolingWindow,
                                                            const IndexVector_t& poolingStride);

        /**
         * Add an activation layer to the network.
         *
         * \param[in] activation Tag that describes the activation function
         * \param[in] alpha Alpha parameter for the activation function. If the activation function
         *            doesn't use this parameter it is ignored. This parameter is optional and
         *            defaults to 0.
         * \param[in] beta Beta parameter for the activation function. If the
         *            activation function doesn't use this parameter it is ignored. This parameter
         * is optional and defaults to 0.
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>& addActivationLayer(Activation activation,
                                                               data_t alpha = 0, data_t beta = 0);

        /**
         * Add a convolution layer to the network
         *
         * \param[in] weightsDescriptor Descriptor for the layer's weights
         * \param[in] strideVector Vector describing the stride for each spatial dimension
         * \param[in] paddingVector Vector describing padding for each spatial dimension
         * \param[in] initializer The layer's initializer
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>&
            addConvLayer(const DataDescriptor& weightsDescriptor, const IndexVector_t& strideVector,
                         const IndexVector_t& paddingVector,
                         Initializer initializer = Initializer::Uniform);

        /**
         * Add a convolution layer to the network
         *
         * \param[in] numFilters The number of convolution filters
         * \param[in] weightsVector The spatial dimensions of each filter
         * \param[in] strideVector Vector describing the stride for each spatial dimension
         * \param[in] paddingVector Vector describing padding for each spatial dimension
         * \param[in] initializer The layer's initializer
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>&
            addConvLayer(index_t numFilters, const IndexVector_t& weightsVector,
                         const IndexVector_t& strideVector, const IndexVector_t& paddingVector,
                         Initializer initializer = Initializer::Uniform);

        /**
         * Add a convolution layer to the network
         *
         * \param[in] numFilters The number of convolution filters
         * \param[in] weightsVector The spatial dimensions of each filter
         * \param[in] initializer The layer's initializer
         *
         * \note This assumes convolution strides of the filters' sizes and no padding
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>&
            addConvLayer(index_t numFilters, const IndexVector_t& weightsVector,
                         Initializer initializer = Initializer::Uniform);

        SequentialNetwork<data_t, Backend>& addFixedLayer(const JosephsMethod<data_t>& op);

        SequentialNetwork<data_t, Backend>& addLRNLayer(index_t localSize,
                                                        data_t alpha = static_cast<data_t>(1),
                                                        data_t beta = static_cast<data_t>(1),
                                                        data_t k = static_cast<data_t>(1));
        /**
         * Add a softmax layer to the network
         *
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>& addSoftmaxLayer();

        SequentialNetwork<data_t, Backend>& setLoss(Loss loss);

        /**
         * Get the network's output descriptor.
         *
         * \note This function throws if the network contains no layers
         */
        DataDescriptor getOutputDescriptor() const;

        /// Get the network's input descriptor
        DataDescriptor getInputDescriptor() const;

        /// Get the number of layers in the network
        std::size_t getNumberOfLayers() const;

        /// Forward propagate input through all network layers
        void forwardPropagate(const DataContainer<data_t>& input);

        std::vector<data_t> train(const DataContainer<data_t>& input,
                                  const DataContainer<data_t>& label);

        /**
         * Get network output
         *
         * \note This function throws if the output contains no layers
         */
        DataContainer<data_t> getOutput() const;

        void compile();

        std::shared_ptr<Layer<data_t, Backend>> getLayer(const index_t index);

    private:
        /**
         * Add a layer to the network.
         *
         * Add a layer to the network by specifying its type and its constructors arguments expect
         * the input descriptor which is chosen automatically depending on previous layers.
         *
         * \tparam LayerType Type of the layer to add
         * \tparam ArgType Parameter pack
         */
        template <typename LayerType, typename... ArgTypes>
        SequentialNetwork<data_t, Backend>& addLayer(const ArgTypes&... args)
        {
            // If this is the first layer we add, its input descriptor is the networks input
            // descriptor. Otherwise we use the last layer's output descriptor as an input
            // descriptor.
            if (_layerStack.empty()) {
                _layerStack.emplace_back(std::make_shared<LayerType>(*_inputDescriptor, args...));
            } else {
                auto outputDesc = _layerStack.back()->getOutputDescriptor();
                _layerStack.emplace_back(std::make_shared<LayerType>(outputDesc, args...));
            }

            return *this;
        }

        /// The networks input descriptor
        std::unique_ptr<DataDescriptor> _inputDescriptor;

        /// A vector containing all network layer
        std::vector<std::shared_ptr<Layer<data_t, Backend>>> _layerStack;

        /// A pointer to this layer's backend
        std::unique_ptr<BackendNetworkType> _backend = nullptr;

        /// Flag to indicate that an input has been propagated through the network
        bool _isPropagated = false;

        /// The networks loss function
        Loss _loss = Loss::MeanSquareError;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<SequentialNetwork<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSequentialNetwork<data_t>;
        };
    } // namespace detail
} // namespace elsa