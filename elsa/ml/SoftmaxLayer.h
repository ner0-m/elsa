#pragma once

#include "Layer.h"
#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "DnnlSoftmaxLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    class SoftmaxLayer : public Layer<data_t, _BackendTag>
    {
    public:
        using BackendLayerType = typename detail::BackendSelector<SoftmaxLayer>::Type;
        using BaseType = Layer<data_t, _BackendTag>;

        SoftmaxLayer() = default;

        SoftmaxLayer(const DataDescriptor& inputDescriptor);

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
        struct BackendSelector<SoftmaxLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSoftmaxLayer<data_t>;
        };
    } // namespace detail
} // namespace elsa
