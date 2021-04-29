#include "Dense.h"

namespace elsa::ml
{
    template <typename data_t>
    Dense<data_t>::Dense(index_t units, Activation activation, bool useBias,
                         Initializer kernelInitializer, Initializer biasInitializer,
                         const std::string& name)
        : Trainable<data_t>(LayerType::Dense, activation, useBias, kernelInitializer,
                            biasInitializer, name, /* required number of input dims */ 1),
          units_(units)
    {
    }

    template <typename data_t>
    index_t Dense<data_t>::getNumberOfUnits() const
    {
        return units_;
    }

    template <typename data_t>
    void Dense<data_t>::computeOutputDescriptor()
    {
        IndexVector_t dims(1);
        dims << units_;
        this->outputDescriptor_ = VolumeDescriptor(dims).clone();

        this->numberOfTrainableParameters_ =
            units_ * this->getInputDescriptor().getNumberOfCoefficients()
            + (this->useBias_ ? units_ : 0);
    }

    template class Dense<float>;
} // namespace elsa::ml
