#include "Input.h"

namespace elsa::ml
{
    template <typename data_t>
    Input<data_t>::Input(const VolumeDescriptor& inputDescriptor, index_t batchSize,
                         const std::string& name)
        : Layer<data_t>(LayerType::Input, name, Layer<data_t>::AnyNumberOfInputDimensions,
                        /* allowedNumberOfInputs */ 0),
          batchSize_(batchSize)
    {
        this->setInputDescriptor(inputDescriptor);
    }

    template <typename data_t>
    index_t Input<data_t>::getBatchSize() const
    {
        return batchSize_;
    }

    template class Input<float>;
} // namespace elsa::ml
