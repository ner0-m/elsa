#include "Pooling.h"

namespace elsa::ml
{
    template <typename data_t>
    Pooling<data_t>::Pooling(LayerType layerType, index_t poolingDimensions,
                             const IndexVector_t& poolSize, index_t strides, Padding padding,
                             const std::string& name)
        : Layer<data_t>(layerType, name, /* input dims */ static_cast<int>(poolingDimensions) + 1,
                        /* inputs */ 1),
          poolingDimensions_(poolingDimensions),
          poolSize_(poolSize),
          strides_(strides),
          padding_(padding)
    {
    }

    template <typename data_t>
    IndexVector_t Pooling<data_t>::getPoolSize() const
    {
        return poolSize_;
    }

    template <typename data_t>
    index_t Pooling<data_t>::getStrides() const
    {
        return strides_;
    }

    template <typename data_t>
    Padding Pooling<data_t>::getPadding() const
    {
        return padding_;
    }

    template <typename data_t>
    void Pooling<data_t>::computeOutputDescriptor()
    {
        // output = (input - poolSize) / stride + 1
        IndexVector_t dims(poolingDimensions_ + 1);
        for (int idx = 0; idx < poolingDimensions_; ++idx)
            dims[idx] = (this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension()[idx]
                         - poolSize_[idx])
                            / strides_
                        + 1;
        dims(poolingDimensions_) = this->inputDescriptors_.front()
                                       ->getNumberOfCoefficientsPerDimension()[poolingDimensions_];
        this->outputDescriptor_ = VolumeDescriptor(dims).clone();
    }

    template <typename data_t>
    MaxPooling1D<data_t>::MaxPooling1D(index_t poolSize, index_t strides, Padding padding,
                                       const std::string& name)
        : Pooling<data_t>(LayerType::MaxPooling1D, 1, IndexVector_t{{poolSize}}, strides, padding,
                          name)
    {
    }

    template <typename data_t>
    MaxPooling2D<data_t>::MaxPooling2D(const IndexVector_t& poolSize, index_t strides,
                                       Padding padding, const std::string& name)
        : Pooling<data_t>(LayerType::MaxPooling2D, 2, {poolSize}, strides, padding, name)
    {
    }

    template class Pooling<float>;
    template struct MaxPooling1D<float>;
    template struct MaxPooling2D<float>;
} // namespace elsa::ml