#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "TrainableLayer.h"
#include "DnnlDenseLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class DenseLayer final : public TrainableLayer<data_t, _BackendTag>
    {
    public:
        using BaseType = TrainableLayer<data_t, _BackendTag>;
        using BaseType::initializer;

        using BackendLayerType = typename detail::BackendSelector<DenseLayer>::Type;

        /**
         * Construct a convolutional network layer
         *
         * \param[in] inputDescriptor DataDescriptor for the input data in either nchw or nchwd
         *  format
         * \param[in] numNeurons Number of Neurons in the dense layer
         * \param[in] initializer The initializer for the layer's weights and biases. This parameter
         * is optional and defaults to Initializer::Uniform
         */
        DenseLayer(const DataDescriptor& inputDescriptor, int numNeurons,
              Initializer initializer = Initializer::Uniform);

    private:
        using BaseType::_backend;

        /// \copydoc TrainableLayer::_inputDescriptor
        using BaseType::_inputDescriptor;

        /// \copydoc TrainableLayer::_outputDescriptor
        using BaseType::_outputDescriptor;

        /// \copydoc TrainableLayer::_weightsDescriptor
        using BaseType::_weightsDescriptor;

        /// \copydoc TrainableLayer::_biasDescriptor
        using BaseType::_biasDescriptor;

        int _numNeurons;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<DenseLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlDenseLayer<data_t>;
        };
    } // namespace detail
} // namespace elsa
