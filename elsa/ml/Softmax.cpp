#include "Softmax.h"

namespace elsa::ml
{
    template <typename data_t>
    Softmax<data_t>::Softmax(index_t axis, const std::string& name)
        : Layer<data_t>(LayerType::Softmax, name), axis_(axis)
    {
    }

    template <typename data_t>
    index_t Softmax<data_t>::getAxis() const
    {
        return axis_;
    }

    template <typename data_t>
    void Softmax<data_t>::computeOutputDescriptor()
    {
        this->outputDescriptor_ = this->inputDescriptors_.front()->clone();
    }

    template class Softmax<float>;
} // namespace elsa::ml