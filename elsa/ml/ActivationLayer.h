#pragma once

#include "elsaDefines.h"
#include "DataContainer.h"
#include "DataDescriptor.h"
#include "Layer.h"
#include "DnnlActivationLayer.h"

namespace elsa
{
    enum class Activation {
        Abs,
        BoundedRelu,
        Elu,
        Exp,
        Linear,
        Gelu,
        Logistic,
        Relu,
        SoftRelu,
        Sqrt,
        Square,
        Swish,
        Tanh
    };

    template <typename data_t, MlBackend _BackendTag, typename Derived>
    class ActivationLayer : public Layer<data_t, _BackendTag>
    {
    public:
        using BaseType = Layer<data_t, _BackendTag>;
        using BackendLayerType = typename detail::BackendSelector<Derived>::Type;
        static constexpr MlBackend BackendTag = _BackendTag;

        /**
         * Construct an activation layer by specifiying its input descriptor and alpha and beta
         * values.
         *
         * \param[in] inputDescriptor Descriptor for the layer's input
         * \param[in] alpha Alpha parameter in the layer's activation function. This parameter is
         * optional and defaults to 0
         * \param[in] beta Beta parameter in the layer's activation
         * function. This parameter is optional and defaults to 0
         *
         * \note Alpha and beta parameters are ignored if the activation function doesn't use them
         */
        ActivationLayer(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0)
            : Layer<data_t, _BackendTag>(inputDescriptor)
        {
            _outputDescriptor = inputDescriptor.clone();
            _backend = std::make_shared<BackendLayerType>(inputDescriptor, *_outputDescriptor);
            setAlpha(alpha);
            setBeta(beta);
        }

        /// Set alpha parameter in the backend layer
        void setAlpha(data_t alpha) const
        {
            std::static_pointer_cast<BackendLayerType>(_backend)->setAlpha(alpha);
        }

        /// Set beta parameter in the backend layer
        void setBeta(data_t beta) const
        {
            std::static_pointer_cast<BackendLayerType>(_backend)->setBeta(beta);
        }

    protected:
        /// \copydoc Layer::_outputDescriptor
        using BaseType::_outputDescriptor;

        /// \copydoc Layer::_backend
        using BaseType::_backend;
    }; // namespace elsa

    /**
     * An Abs activation layer.
     *
     * This layer perfoms element-wise abs on its input, i.e.,
     *
     * `abs(x) = |x|`
     *
     * \tparam data_t Type for all coefficients used in the layer. This parameter is optional and
     * defaults to real_t.
     * \tparam _BackendTag Tag to specify the layer's backend. This parameter is
     * optional an defaults to MlBackend::Dnnl.
     */
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct Abs final : public ActivationLayer<data_t, _BackendTag, Abs<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, Abs<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;

        /**
         * Construct an Abs activation layer by specifiying its input descriptor and alpha and beta
         * values.
         *
         * \param[in] inputDescriptor Descriptor for the layer's input.
         * \param[in] alpha Alpha parameter in the layer's activation function. This parameter is
         * optional and defaults to 0.
         * \param[in] beta Beta parameter in the layer's activation
         * function. This parameter is optional and defaults to 0.
         *
         * \note Both, alpha and beta parameters are ignored by this layer.
         */
        Abs(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Abs<data_t, MlBackend::Dnnl>> {
            using Type = DnnlAbs<data_t>;
        };
    } // namespace detail

    /**
     * A BoundesRelu activation layer.
     *
     * This layer perfoms element-wise boundes-relu on its input, i.e.,
     *
     * `boundedRelu(x) = |x|`
     *
     * \tparam data_t Type for all coefficients used in the layer. This parameter is optional and
     * defaults to real_t.
     * \tparam _BackendTag Tag to specify the layer's backend. This parameter is
     * optional an defaults to MlBackend::Dnnl.
     */
    template <typename data_t = real_t, MlBackend _BackendTag = MlBackend::Dnnl>
    struct BoundedRelu final
        : public ActivationLayer<data_t, _BackendTag, BoundedRelu<data_t, _BackendTag>> {
        using BaseType = ActivationLayer<data_t, _BackendTag, BoundedRelu<data_t, _BackendTag>>;
        using BackendLayerType = typename BaseType::BackendLayerType;
        static constexpr MlBackend BackendTag = _BackendTag;
        BoundedRelu(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Elu(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Exp(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Linear(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Gelu(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Logistic(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Relu(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        SoftRelu(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Sqrt(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Square(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Swish(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
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
        Tanh(const DataDescriptor& inputDescriptor, data_t alpha = 0, data_t beta = 0);
    };

    namespace detail
    {
        template <typename data_t>
        struct BackendSelector<Tanh<data_t, MlBackend::Dnnl>> {
            using Type = DnnlTanh<data_t>;
        };
    } // namespace detail

} // namespace elsa