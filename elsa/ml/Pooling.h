#pragma once

#include "Layer.h"
#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DnnlPooling.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class Pooling final : public Layer<data_t, _BackendTag>
    {
    public:
        using BackendLayerType = typename detail::BackendSelector<Pooling>::Type;
        using BaseType = Layer<data_t, _BackendTag>;

        Pooling() = default;

        /**
         * Constructor for a max pooling layer
         *
         * \param[in] inputDescriptor DataDescriptor for the layer's input
         * \param[in] poolingWindow
         * \param[in] poolingStride
         */
        Pooling(const DataDescriptor& inputDescriptor, const IndexVector_t& poolingWindow,
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
        struct BackendSelector<Pooling<data_t, MlBackend::Dnnl>> {
            using Type = DnnlPooling<data_t>;
        };
    } // namespace detail

} // namespace elsa