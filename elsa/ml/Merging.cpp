#include "Merging.h"

namespace elsa::ml
{
    template <typename data_t>
    Merging<data_t>::Merging(LayerType layerType, std::initializer_list<Layer<data_t>*> inputs,
                             const std::string& name)
        : Layer<data_t>(layerType, name)
    {
        this->setInput(inputs);
    }

    template <typename data_t>
    bool Merging<data_t>::canMerge() const
    {
        return true;
    }

    template <typename data_t>
    Sum<data_t>::Sum(std::initializer_list<Layer<data_t>*> inputs, const std::string& name)
        : Merging<data_t>(LayerType::Sum, inputs, name)
    {
    }

    template <typename data_t>
    void Sum<data_t>::computeOutputDescriptor()
    {
        if (std::adjacent_find(this->inputDescriptors_.begin(), this->inputDescriptors_.end(),
                               [](const auto& a, const auto& b) { return *a != *b; })
            != this->inputDescriptors_.end()) {
            throw std::invalid_argument("All inputs for Sum layer must have the same shape");
        }

        // At this point we are sure that all input descriptors match. Since we just
        // compute the coeff-wise sum it's enough to take just one of the input
        // descriptors as output descriptor
        this->outputDescriptor_ = this->inputDescriptors_.front()->clone();
    }

    template <typename data_t>
    Concatenate<data_t>::Concatenate(index_t axis, std::initializer_list<Layer<data_t>*> inputs,
                                     const std::string& name)
        : Merging<data_t>(LayerType::Concatenate, inputs, name), axis_(axis)
    {
    }

    template <typename data_t>
    void Concatenate<data_t>::computeOutputDescriptor()
    {

        index_t concatDim = 0;
        for (const auto& in : this->inputDescriptors_) {
            concatDim += in->getNumberOfCoefficientsPerDimension()[axis_];
        }
        IndexVector_t dims(this->inputDescriptors_.front()->getNumberOfDimensions());
        for (int i = 0; i < this->inputDescriptors_.front()->getNumberOfDimensions(); ++i)
            dims[i] = this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension()[i];
        dims[axis_] = concatDim;
        this->outputDescriptor_ = VolumeDescriptor(dims).clone();
    }

    template class Merging<float>;
    template class Sum<float>;
    template class Concatenate<float>;
} // namespace elsa::ml
