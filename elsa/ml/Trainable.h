#pragma once

#include <string>

#include "Common.h"
#include "Layer.h"
#include "Activation.h"

namespace elsa::ml
{
    template <typename data_t>
    class Trainable : public Layer<data_t>
    {
    public:
        /// return the activation
        Activation getActivation() const;

        /// return true if the layer uses a bias, false otherwise
        bool useBias() const;

        /// return the kernel initializer
        Initializer getKernelInitializer() const;

        /// return the bias initializer
        Initializer getBiasInitializer() const;

    protected:
        Trainable(LayerType layerType, Activation activation, bool useBias,
                  Initializer kernelInitializer, Initializer biasInitializer,
                  const std::string& name, int requiredNumberOfDimensions);

        bool useBias_;

    private:
        Activation activation_;

        Initializer kernelInitializer_;
        Initializer biasInitializer_;
    };
} // namespace elsa::ml