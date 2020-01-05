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
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>&
            addDenseLayer(int numNeurons, Initializer initializer = Initializer::Uniform);

        /**
         * Add a pooling layer to the network.
         *
         * \param[in] poolingWindow Pooling window of the pooling layer
         * \param[in] poonlingStide Pooling stride of the pooling layer
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
         * \return A reference to the network to allow method chaining
         */
        SequentialNetwork<data_t, Backend>& addActivationLayer(Activation activation,
                                                               data_t alpha = 0, data_t beta = 0);

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

        void forwardPropagate(const DataContainer<data_t>& input);

        DataContainer<data_t> getOutput() const;

        void compile();

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
                _layerStack.emplace_back(LayerType(*_inputDescriptor, args...));
            } else {
                auto outputDesc = _layerStack.back().getOutputDescriptor();
                _layerStack.emplace_back(LayerType(outputDesc, args...));
            }

            return *this;
        }

        std::unique_ptr<DataDescriptor> _inputDescriptor;
        std::vector<Layer<data_t, Backend>> _layerStack;
        std::unique_ptr<BackendNetworkType> _backend = nullptr;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<SequentialNetwork<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSequentialNetwork<data_t>;
        };
    } // namespace detail
} // namespace elsa