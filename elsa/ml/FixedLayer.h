#pragma once

#include "elsaDefines.h"
#include "Geometry.h"
#include "JosephsMethod.h"
#include "Layer.h"
#include "DnnlFixedLayer.h"

namespace elsa
{
    template <typename data_t = real_t, MlBackend Backend = MlBackend::Dnnl>
    class FixedLayer : public Layer<data_t, Backend>
    {
    public:
        using BaseType = Layer<data_t, Backend>;

        using BackendLayerType = typename detail::BackendSelector<FixedLayer>::Type;

        FixedLayer() = default;

        FixedLayer(const DataDescriptor& inputDescriptor, const JosephsMethod<data_t>& op);

        bool isOperator() const;

    private:
        using BaseType::_backend;

        using BaseType::_outputDescriptor;
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<FixedLayer<data_t, MlBackend::Dnnl>> {
            using Type = DnnlFixedLayer<data_t>;
        };
    } // namespace detail
} // namespace elsa