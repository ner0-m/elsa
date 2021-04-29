#include "Trainable.h"

namespace elsa::ml
{
    template <typename data_t>
    Trainable<data_t>::Trainable(LayerType layerType, Activation activation, bool useBias,
                                 Initializer kernelInitializer, Initializer biasInitializer,
                                 const std::string& name, int requiredNumberOfDimensions)
        : Layer<data_t>(layerType, name, requiredNumberOfDimensions,
                        /* allowed number of inputs */ 1,
                        /* is trainable */ true),
          useBias_(useBias),
          activation_(activation),
          kernelInitializer_(kernelInitializer),
          biasInitializer_(biasInitializer)
    {
    }

    template <typename data_t>
    Activation Trainable<data_t>::getActivation() const
    {
        return activation_;
    }

    template <typename data_t>
    bool Trainable<data_t>::useBias() const
    {
        return useBias_;
    }

    template <typename data_t>
    Initializer Trainable<data_t>::getKernelInitializer() const
    {
        return kernelInitializer_;
    }

    template <typename data_t>
    Initializer Trainable<data_t>::getBiasInitializer() const
    {
        return biasInitializer_;
    }

    template class Trainable<float>;
} // namespace elsa::ml