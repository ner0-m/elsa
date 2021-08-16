#include "Conv.h"
#include "TypeCasts.hpp"

namespace elsa::ml
{
    template <typename data_t>
    ConvBase<data_t>::ConvBase(LayerType layerType, index_t numberOfFilters,
                               const VolumeDescriptor& filterDescriptor, Activation activation,
                               index_t strides, Padding padding, bool useBias,
                               Initializer kernelInitializer, Initializer biasInitializer,
                               const std::string& name, int requiredNumberOfDimensions)
        : Trainable<data_t>(layerType, activation, useBias, kernelInitializer, biasInitializer,
                            name, requiredNumberOfDimensions),
          numberOfFilters_(numberOfFilters),
          filterDescriptor_(filterDescriptor.clone()),
          strides_(strides),
          padding_(padding)
    {
    }

    template <typename data_t>
    index_t ConvBase<data_t>::getNumberOfFilters() const
    {
        return numberOfFilters_;
    }

    template <typename data_t>
    VolumeDescriptor ConvBase<data_t>::getFilterDescriptor() const
    {
        if (!filterDescriptor_)
            throw std::logic_error("Filter descriptor not set");

        // Downcast to VolumeDescriptor
        return downcast<VolumeDescriptor>(*filterDescriptor_);
    }

    template <typename data_t>
    index_t ConvBase<data_t>::getStrides() const
    {
        return strides_;
    }

    template <typename data_t>
    Padding ConvBase<data_t>::getPadding() const
    {
        return padding_;
    }

    template <typename data_t>
    Conv<data_t>::Conv(index_t convolutionDimensions, index_t numberOfFilters,
                       const VolumeDescriptor& filterDescriptor, Activation activation,
                       index_t strides, Padding padding, bool useBias,
                       Initializer kernelInitializer, Initializer biasInitializer,
                       const std::string& name)
        : ConvBase<data_t>(
            convolutionDimensions == 1
                ? LayerType::Conv1D
                : (convolutionDimensions == 2 ? LayerType::Conv2D : LayerType::Conv3D),
            numberOfFilters, filterDescriptor, activation, strides, padding, useBias,
            kernelInitializer, biasInitializer, name,
            /* required number of input dims */ static_cast<int>(convolutionDimensions) + 1),
          convolutionDimensions_(convolutionDimensions)
    {
    }

    template <typename data_t>
    IndexVector_t Conv<data_t>::getPaddingSizes() const
    {
        // We pad spatial dimensions only
        IndexVector_t paddingSize(asUnsigned(this->convolutionDimensions_));

        switch (this->getPadding()) {
            // Valid padding means no padding at all
            case Padding::Valid:
                paddingSize.setZero();
                break;
            // Same padding means we pad the input such that the output has unchanges spatial
            // dimensions. In theory we wouldn't need to do anything here since we already
            // know the dimensions of the output descriptor (same as the ones of the input
            // descriptor), however, the backend won't understand magic terms such as *Valid* or
            // *Same* and needs bare padding numbers.
            //
            // Since the spatial output dimensions of a convolution operations
            // calculates as
            //    o = (i-p+k)/s+1
            // we calculate same padding as
            //  p = s(i-1)-1-k
            case Padding::Same:
                for (int dim = 0; dim < paddingSize.size(); ++dim) {
                    const index_t inputDim =
                        this->getInputDescriptor().getNumberOfCoefficientsPerDimension()[dim];
                    const index_t kernelDim =
                        this->getFilterDescriptor().getNumberOfCoefficientsPerDimension()[dim];
                    paddingSize[dim] = this->getStrides() * (inputDim - 1) - inputDim + kernelDim;
                }
                break;
        }
        return paddingSize;
    }

    template <typename data_t>
    void Conv<data_t>::computeOutputDescriptor()
    {
        if (this->inputDescriptors_.size() != 1) {
            throw std::invalid_argument(
                "Cannot compute output descriptor without input descriptor");
        }

        if (this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension().tail(1)(0)
            != this->filterDescriptor_->getNumberOfCoefficientsPerDimension().tail(1)(0))
            throw std::invalid_argument("Input and filter channels must match");

        IndexVector_t dims(convolutionDimensions_ + 1);
        auto padding = getPaddingSizes();
        for (index_t idx = 0; idx < convolutionDimensions_; ++idx) {
            // output = (input - kernel + 2 * padding) / stride + 1
            dims[idx] = (this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension()[idx]
                         - this->filterDescriptor_->getNumberOfCoefficientsPerDimension()[idx]
                         + padding[idx])
                            / this->strides_
                        + 1;
        }
        dims(convolutionDimensions_) = this->numberOfFilters_;
        VolumeDescriptor desc(dims);
        this->outputDescriptor_ = desc.clone();
        this->numberOfTrainableParameters_ =
            this->getNumberOfFilters() * this->getFilterDescriptor().getNumberOfCoefficients()
            + (this->useBias_ ? this->getNumberOfFilters() : 0);
    }

    template <typename data_t>
    Conv1D<data_t>::Conv1D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
                           Activation activation, index_t strides, Padding padding, bool useBias,
                           Initializer kernelInitializer, Initializer biasInitializer,
                           const std::string& name)
        : Conv<data_t>(1, numberOfFilters, filterDescriptor, activation, strides, padding, useBias,
                       kernelInitializer, biasInitializer, name)
    {
    }

    template <typename data_t>
    Conv1D<data_t>::Conv1D(index_t numberOfFilters, const std::array<index_t, 2>& filterSize,
                           Activation activation, index_t strides, Padding padding, bool useBias,
                           Initializer kernelInitializer, Initializer biasInitializer,
                           const std::string& name)
        : Conv<data_t>(1, numberOfFilters,
                       VolumeDescriptor(IndexVector_t{
                           {/* width */ filterSize[0], /* channels */ filterSize[1]}}),
                       activation, strides, padding, useBias, kernelInitializer, biasInitializer,
                       name)
    {
    }

    template <typename data_t>
    Conv2D<data_t>::Conv2D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
                           Activation activation, index_t strides, Padding padding, bool useBias,
                           Initializer kernelInitializer, Initializer biasInitializer,
                           const std::string& name)
        : Conv<data_t>(2, numberOfFilters, filterDescriptor, activation, strides, padding, useBias,
                       kernelInitializer, biasInitializer, name)
    {
    }

    template <typename data_t>
    Conv2D<data_t>::Conv2D(index_t numberOfFilters, const std::array<index_t, 3>& filterSize,
                           Activation activation, index_t strides, Padding padding, bool useBias,
                           Initializer kernelInitializer, Initializer biasInitializer,
                           const std::string& name)
        : Conv<data_t>(
            2, numberOfFilters,
            VolumeDescriptor(IndexVector_t{{/* width */ filterSize[0], /* height */ filterSize[1],
                                            /* channels */ filterSize[2]}}),
            activation, strides, padding, useBias, kernelInitializer, biasInitializer, name)
    {
    }

    template <typename data_t>
    Conv3D<data_t>::Conv3D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
                           Activation activation, index_t strides, Padding padding, bool useBias,
                           Initializer kernelInitializer, Initializer biasInitializer,
                           const std::string& name)
        : Conv<data_t>(3, numberOfFilters, filterDescriptor, activation, strides, padding, useBias,
                       kernelInitializer, biasInitializer, name)
    {
    }

    template <typename data_t>
    ConvTranspose<data_t>::ConvTranspose(index_t convolutionDimensions, index_t numberOfFilters,
                                         const VolumeDescriptor& filterDescriptor,
                                         Activation activation, index_t strides, Padding padding,
                                         bool useBias, Initializer kernelInitializer,
                                         Initializer biasInitializer, const std::string& name)
        : ConvBase<data_t>(
            convolutionDimensions == 2 ? LayerType::Conv2DTranspose : LayerType::Conv3DTranspose,
            numberOfFilters, filterDescriptor, activation, strides, padding, useBias,
            kernelInitializer, biasInitializer, name,
            /* required number of input dims */ static_cast<int>(convolutionDimensions) + 1),
          convolutionDimensions_(convolutionDimensions)
    {
    }

    template <typename data_t>
    void ConvTranspose<data_t>::computeOutputDescriptor()
    {
        // Spatial dimension + channels
        IndexVector_t dims(convolutionDimensions_ + 1);

        // using (width x height x channels) we get

        // TODO(tellenbach): Add padding
        for (index_t idx = 0; idx < convolutionDimensions_; ++idx) {
            dims[idx] =
                this->strides_
                    * (this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension()[idx]
                       - 1)
                + this->filterDescriptor_
                      ->getNumberOfCoefficientsPerDimension()[idx]; //- 2 * padding[idx];
        }
        dims(convolutionDimensions_) = this->numberOfFilters_;

        this->outputDescriptor_ = VolumeDescriptor(dims).clone();
    }

    template <typename data_t>
    Conv2DTranspose<data_t>::Conv2DTranspose(index_t numberOfFilters,
                                             const VolumeDescriptor& filterDescriptor,
                                             Activation activation, index_t strides,
                                             Padding padding, bool useBias,
                                             Initializer kernelInitializer,
                                             Initializer biasInitializer, const std::string& name)
        : ConvTranspose<data_t>(2, numberOfFilters, filterDescriptor, activation, strides, padding,
                                useBias, kernelInitializer, biasInitializer, name)
    {
    }

    template class ConvBase<float>;
    template class Conv<float>;
    template struct Conv1D<float>;
    template struct Conv2D<float>;
    template struct Conv3D<float>;
    template class ConvTranspose<float>;
    template struct Conv2DTranspose<float>;

} // namespace elsa::ml
