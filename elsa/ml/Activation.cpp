#include "Activation.h"

namespace elsa
{
#define ELSA_ACTIVATION_LAYER_IMPL(name)                                                   \
    template <typename data_t, MlBackend _BackendTag>                                      \
    name<data_t, _BackendTag>::name(const DataDescriptor& inputDescriptor)                 \
        : ActivationLayer<data_t, _BackendTag, name<data_t, _BackendTag>>(inputDescriptor) \
    {                                                                                      \
    }

    ELSA_ACTIVATION_LAYER_IMPL(Abs)
    ELSA_ACTIVATION_LAYER_IMPL(BoundedRelu)
    ELSA_ACTIVATION_LAYER_IMPL(Elu)
    ELSA_ACTIVATION_LAYER_IMPL(Exp)
    ELSA_ACTIVATION_LAYER_IMPL(Linear)
    ELSA_ACTIVATION_LAYER_IMPL(Gelu)
    ELSA_ACTIVATION_LAYER_IMPL(Logistic)
    ELSA_ACTIVATION_LAYER_IMPL(Relu)
    ELSA_ACTIVATION_LAYER_IMPL(SoftRelu)
    ELSA_ACTIVATION_LAYER_IMPL(Sqrt)
    ELSA_ACTIVATION_LAYER_IMPL(Square)
    ELSA_ACTIVATION_LAYER_IMPL(Swish)
    ELSA_ACTIVATION_LAYER_IMPL(Tanh)

#undef ELSA_ACTIVATION_LAYER_IMPL

#define ELSA_ACTIVATION_LAYER_INSTANTIATION(name) template struct name<float, MlBackend::Dnnl>;

    ELSA_ACTIVATION_LAYER_INSTANTIATION(Abs)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(BoundedRelu)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Elu)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Exp)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Linear)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Gelu)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Logistic)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Relu)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(SoftRelu)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Sqrt)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Square)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Swish)
    ELSA_ACTIVATION_LAYER_INSTANTIATION(Tanh)

#undef ELSA_ACTIVATION_LAYER_INSTANTIATION

} // namespace elsa
