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

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Abs final : public ActivationLayer<data_t, _BackendTag, Abs<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Abs<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Abs(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Abs<data_t, MlBackend::Dnnl>> {
            using Type = DnnlAbs<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct BoundedRelu final
        : public ActivationLayer<data_t, _BackendTag, BoundedRelu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, BoundedRelu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit BoundedRelu(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<BoundedRelu<data_t, MlBackend::Dnnl>> {
            using Type = DnnlBoundedRelu<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Elu final : public ActivationLayer<data_t, _BackendTag, Elu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Elu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Elu(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Elu<data_t, MlBackend::Dnnl>> {
            using Type = DnnlElu<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Exp final : public ActivationLayer<data_t, _BackendTag, Exp<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Exp<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Exp(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Exp<data_t, MlBackend::Dnnl>> {
            using Type = DnnlExp<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Linear final : public ActivationLayer<data_t, _BackendTag, Linear<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Linear<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Linear(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Linear<data_t, MlBackend::Dnnl>> {
            using Type = DnnlLinear<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Gelu final : public ActivationLayer<data_t, _BackendTag, Gelu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Gelu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Gelu(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Gelu<data_t, MlBackend::Dnnl>> {
            using Type = DnnlGelu<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Logistic final
        : public ActivationLayer<data_t, _BackendTag, Logistic<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Logistic<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Logistic(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Logistic<data_t, MlBackend::Dnnl>> {
            using Type = DnnlLogistic<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Relu final : public ActivationLayer<data_t, _BackendTag, Relu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Relu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Relu(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Relu<data_t, MlBackend::Dnnl>> {
            using Type = DnnlRelu<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct SoftRelu final
        : public ActivationLayer<data_t, _BackendTag, SoftRelu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, SoftRelu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit SoftRelu(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<SoftRelu<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSoftRelu<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Sqrt final : public ActivationLayer<data_t, _BackendTag, Sqrt<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Sqrt<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Sqrt(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Sqrt<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSqrt<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Square final : public ActivationLayer<data_t, _BackendTag, Square<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Square<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Square(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Square<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSquare<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Swish final : public ActivationLayer<data_t, _BackendTag, Swish<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Swish<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Swish(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Swish<data_t, MlBackend::Dnnl>> {
            using Type = DnnlSwish<data_t>;
        };
    } // namespace detail

    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Tanh final : public ActivationLayer<data_t, _BackendTag, Tanh<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Tanh<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        explicit Tanh(const DataDescriptor& inputDescriptor);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Tanh<data_t, MlBackend::Dnnl>> {
            using Type = DnnlTanh<data_t>;
        };
    } // namespace detail
} // namespace elsa