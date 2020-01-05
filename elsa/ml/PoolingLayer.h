#pragma once

#include "Layer.h"
#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DnnlPoolingLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class PoolingLayer final : public Layer<data_t, _BackendTag>
    {
    public:
        using BackendLayerType = typename detail::BackendSelector<PoolingLayer>::Type;
        using BaseType = Layer<data_t, _BackendTag>;

        PoolingLayer() = default;

        /**
         * Constructor for a max pooling layer
         *
         * \param[in] inputDescriptor DataDescriptor for the layer's input
         * \param[in] poolingWindow
         * \param[in] poolingStride
         */
        PoolingLayer(const DataDescriptor& inputDescriptor, const IndexVector_t& poolingWindow,
                     const IndexVector_t& poolingStride);

    private:
        /// \copydoc Layer::_backend
        using BaseType::_backend;

        /// \copydoc Layer::_inputDescriptor
        using BaseType::_inputDescriptor;

        /// \copydoc Layer::_outputDescriptor
        using BaseType::_outputDescriptor;

        /// The layer's pooling window
        IndexVector_t _poolingWindow;

        /// The layer's pooling strides
        IndexVector_t _poolingStride;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<PoolingLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlPoolingLayer<data_t>;
        };
    } // namespace detail

} // namespace elsa