#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Layer.h"
#include "DnnlActivation.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag, typename Derived>
    class ActivationLayer : public Layer<data_t, _BackendTag>
    {
    public:
        using BaseType = Layer<data_t, _BackendTag>;
        using BackendLayerType = typename detail::BackendSelector<Derived>::Type;
        static constexpr MlBackend BackendTag = _BackendTag;

        explicit ActivationLayer(const DataDescriptor& inputDescriptor)
            : Layer<data_t, _BackendTag>(inputDescriptor)
        {
            _outputDescriptor = inputDescriptor.clone();
            _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor);
        }

        void setAlpha(data_t alpha) const
        {
            std::static_pointer_cast<BackendLayerType>(_backend)->setAlpha(alpha);
        }

        void setBeta(data_t alpha) const
        {
            std::static_pointer_cast<BackendLayerType>(_backend)->setAlpha(alpha);
        }

    protected:
        using BaseType::_outputDescriptor;
        using BaseType::_backend;
    }; // namespace elsa

#define ELSA_ACTIVATION_LAYER_DECLARATION(name)                                                  \
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>                 \
    struct name final : public ActivationLayer<data_t, _BackendTag, name<data_t, _BackendTag>> { \
        using BaseType = ActivationLayer<data_t, _BackendTag, name<data_t, _BackendTag>>;        \
        using BackendLayerType = typename BaseType::BackendLayerType;                            \
        static constexpr MlBackend BackendTag = _BackendTag;                                     \
        explicit name(const DataDescriptor& inputDescriptor);                                    \
    };                                                                                           \
    namespace detail                                                                             \
    {                                                                                            \
        template <typename data_t>                                                               \
        struct BackendSelector<name<data_t, MlBackend::Dnnl>> {                                  \
            using Type = Dnnl##name<data_t>;                                                     \
        };                                                                                       \
    }

    ELSA_ACTIVATION_LAYER_DECLARATION(Abs)
    ELSA_ACTIVATION_LAYER_DECLARATION(BoundedRelu)
    ELSA_ACTIVATION_LAYER_DECLARATION(Elu)
    ELSA_ACTIVATION_LAYER_DECLARATION(Exp)
    ELSA_ACTIVATION_LAYER_DECLARATION(Linear)
    ELSA_ACTIVATION_LAYER_DECLARATION(Gelu)
    ELSA_ACTIVATION_LAYER_DECLARATION(Logistic)
    ELSA_ACTIVATION_LAYER_DECLARATION(Relu)
    ELSA_ACTIVATION_LAYER_DECLARATION(SoftRelu)
    ELSA_ACTIVATION_LAYER_DECLARATION(Sqrt)
    ELSA_ACTIVATION_LAYER_DECLARATION(Square)
    ELSA_ACTIVATION_LAYER_DECLARATION(Swish)
    ELSA_ACTIVATION_LAYER_DECLARATION(Tanh)

#undef ELSA_ACTIVATION_LAYER_DECLARATION

} // namespace elsa