#include "Activation.h"

namespace elsa::ml
{
    template <typename data_t>
    ActivationBase<data_t>::ActivationBase(Activation activation, const std::string& name)
        : Layer<data_t>(LayerType::Activation, name, Layer<data_t>::AnyNumberOfInputDimensions,
                        /* allowed number of inputs */ 1),
          activation_(activation)
    {
    }

    template <typename data_t>
    Activation ActivationBase<data_t>::getActivation() const
    {
        return activation_;
    }

    template <typename data_t>
    Sigmoid<data_t>::Sigmoid(const std::string& name)
        : ActivationBase<data_t>(Activation::Sigmoid, name)
    {
    }

    template <typename data_t>
    Relu<data_t>::Relu(const std::string& name) : ActivationBase<data_t>(Activation::Relu, name)
    {
    }

    template <typename data_t>
    Tanh<data_t>::Tanh(const std::string& name) : ActivationBase<data_t>(Activation::Tanh, name)
    {
    }

    template <typename data_t>
    ClippedRelu<data_t>::ClippedRelu(const std::string& name)
        : ActivationBase<data_t>(Activation::ClippedRelu, name)
    {
    }

    template <typename data_t>
    Elu<data_t>::Elu(const std::string& name) : ActivationBase<data_t>(Activation::Elu, name)
    {
    }

    template <typename data_t>
    IdentityActivation<data_t>::IdentityActivation(const std::string& name)
        : ActivationBase<data_t>(Activation::Identity, name)
    {
    }

    template class ActivationBase<float>;
    template struct Sigmoid<float>;
    template struct Relu<float>;
    template struct Tanh<float>;
    template struct ClippedRelu<float>;
    template struct Elu<float>;
    template struct IdentityActivation<float>;
} // namespace elsa::ml
