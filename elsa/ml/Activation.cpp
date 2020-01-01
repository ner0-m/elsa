#include "Activation.h"

namespace elsa
{
    template <typename data_t, MlBackend _BackendTag>
    Abs<data_t, _BackendTag>::Abs(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Abs<data_t, _BackendTag>>(inputDescriptor)
    {
    }

    template <typename data_t, MlBackend _BackendTag>
    BoundedRelu<data_t, _BackendTag>::BoundedRelu(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, BoundedRelu<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Elu<data_t, _BackendTag>::Elu(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Elu<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Exp<data_t, _BackendTag>::Exp(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Exp<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Linear<data_t, _BackendTag>::Linear(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Linear<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Gelu<data_t, _BackendTag>::Gelu(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Gelu<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Logistic<data_t, _BackendTag>::Logistic(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Logistic<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Relu<data_t, _BackendTag>::Relu(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Relu<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    SoftRelu<data_t, _BackendTag>::SoftRelu(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, SoftRelu<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Sqrt<data_t, _BackendTag>::Sqrt(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Sqrt<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Square<data_t, _BackendTag>::Square(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Square<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Swish<data_t, _BackendTag>::Swish(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Swish<data_t, _BackendTag>>(inputDescriptor)
    {
    }
    template <typename data_t, MlBackend _BackendTag>
    Tanh<data_t, _BackendTag>::Tanh(const DataDescriptor& inputDescriptor)
        : ActivationLayer<data_t, _BackendTag, Tanh<data_t, _BackendTag>>(inputDescriptor)
    {
    }

    template struct Abs<float, MlBackend::Dnnl>;
    template struct BoundedRelu<float, MlBackend::Dnnl>;
    template struct Elu<float, MlBackend::Dnnl>;
    template struct Exp<float, MlBackend::Dnnl>;
    template struct Linear<float, MlBackend::Dnnl>;
    template struct Gelu<float, MlBackend::Dnnl>;
    template struct Logistic<float, MlBackend::Dnnl>;
    template struct Relu<float, MlBackend::Dnnl>;
    template struct SoftRelu<float, MlBackend::Dnnl>;
    template struct Sqrt<float, MlBackend::Dnnl>;
    template struct Square<float, MlBackend::Dnnl>;
    template struct Swish<float, MlBackend::Dnnl>;
    template struct Tanh<float, MlBackend::Dnnl>;

} // namespace elsa
