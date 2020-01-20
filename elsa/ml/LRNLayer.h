#pragma once

#include "Layer.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "DnnlLRNLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class LRNLayer : public Layer<data_t, _BackendTag>
    {
    public:
        using BackendLayerType = typename detail::BackendSelector<LRNLayer>::Type;

        using BaseType = Layer<data_t, _BackendTag>;

        LRNLayer() = default;

        LRNLayer(const DataDescriptor& inputDescriptor, index_t localSize,
                 data_t alpha = static_cast<data_t>(1), data_t beta = static_cast<data_t>(1),
                 data_t k = static_cast<data_t>(1));

    private:
        /// \copydoc Layer::_backend
        using BaseType::_backend;

        /// \copydoc Layer::_inputDescriptor
        using BaseType::_inputDescriptor;

        /// \copydoc Layer::_outputDescriptor
        using BaseType::_outputDescriptor;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<LRNLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlLRNLayer<data_t>;
        };
    } // namespace detail
} // namespace elsa
