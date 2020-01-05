#pragma once

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "TrainableLayer.h"
#include "DnnlConvLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class ConvLayer final : public TrainableLayer<data_t, _BackendTag>
    {
    public:
        using BaseType = TrainableLayer<data_t, _BackendTag>;
        using BaseType::initializer;

        using BackendLayerType = typename detail::BackendSelector<ConvLayer>::Type;
        /**
         * Construct a convolutional network layer
         *
         * \param[in] inputDescriptor DataDescriptor for the input data in either nchw or nchwd
         *  format
         * \param[in] weightsDescriptor DataDescriptor for the convolution filters.
         * \param[in] strideVector Vector containing convolution strides for each spatial dimension
         * \param[in] paddingVector Vector containing padding for each spatial dimension
         */
        ConvLayer(const DataDescriptor& inputDescriptor, const DataDescriptor& weightsDescriptor,
                  const IndexVector_t& strideVector, const IndexVector_t& paddingVector,
                  Initializer initializer = Initializer::Uniform);

        // /// \copydoc Layer::forwardPropagate
        // virtual void forwardPropagate(const DataContainer<data_t>& input) override;

        // /// \copydoc Layer::backwardPropagate
        // virtual void backwardPropagate(const DataContainer<data_t>& input) override;

        // /// \copydoc TrainableLayer::updateTrainableParameters
        // virtual void updateTrainableParameters() override;

    private:
        using BaseType::_backend;

        /// \copydoc TrainableLayer::_inputDescriptor
        using BaseType::_inputDescriptor;

        /// \copydoc TrainableLayer::_outputDescriptor
        using BaseType::_outputDescriptor;

        /// \copydoc TrainableLayer::_weightsDescriptor
        using BaseType::_weightsDescriptor;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<ConvLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlConvLayer<data_t>;
        };
    } // namespace detail
} // namespace elsa
