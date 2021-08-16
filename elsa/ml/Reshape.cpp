#include "Reshape.h"

namespace elsa::ml
{
    template <typename data_t>
    Reshape<data_t>::Reshape(const VolumeDescriptor& targetShape, const std::string& name)
        : Layer<data_t>(LayerType::Reshape, name)
    {

        this->outputDescriptor_ = targetShape.clone();
    }

    template <typename data_t>
    void Reshape<data_t>::computeOutputDescriptor()
    {
        if (this->outputDescriptor_->getNumberOfCoefficients()
            != this->inputDescriptors_.front()->getNumberOfCoefficients())
            throw std::invalid_argument(
                "Descriptors of input and reshaping target must be of same size");
    }

    template <typename data_t>
    Flatten<data_t>::Flatten(const std::string& name) : Layer<data_t>(LayerType::Flatten, name)
    {
    }

    template <typename data_t>
    void Flatten<data_t>::computeOutputDescriptor()
    {
        IndexVector_t dims(1);
        dims[0] = this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension().prod();
        this->outputDescriptor_ = VolumeDescriptor(dims).clone();
    }

    template <typename data_t, LayerType layerType, index_t upSamplingDimensions>
    UpSampling<data_t, layerType, upSamplingDimensions>::UpSampling(
        const std::array<index_t, upSamplingDimensions>& size, Interpolation interpolation,
        const std::string& name)
        : Layer<data_t>(layerType, name), size_(size), interpolation_(interpolation)
    {
    }

    template <typename data_t, LayerType layerType, index_t upSamplingDimensions>
    void UpSampling<data_t, layerType, upSamplingDimensions>::computeOutputDescriptor()
    {
        IndexVector_t dims = this->inputDescriptors_.front()->getNumberOfCoefficientsPerDimension();

        // spatial dims that get upsampled
        if constexpr (upSamplingDimensions >= 1) {
            dims[0] *= size_[0];
        }

        if constexpr (upSamplingDimensions >= 2) {
            dims[1] *= size_[1];
        }

        if constexpr (upSamplingDimensions == 3) {
            dims[2] *= size_[2];
        }

        this->outputDescriptor_ = VolumeDescriptor(dims).clone();
    }

    template <typename data_t, LayerType layerType, index_t upSamplingDimensions>
    Interpolation UpSampling<data_t, layerType, upSamplingDimensions>::getInterpolation() const
    {
        return interpolation_;
    }

    template <typename data_t>
    UpSampling1D<data_t>::UpSampling1D(const std::array<index_t, 1>& size,
                                       Interpolation interpolation, const std::string& name)
        : UpSampling<data_t, LayerType::UpSampling1D, 1>(size, interpolation, name)
    {
    }

    template <typename data_t>
    UpSampling2D<data_t>::UpSampling2D(const std::array<index_t, 2>& size,
                                       Interpolation interpolation, const std::string& name)
        : UpSampling<data_t, LayerType::UpSampling2D, 2>(size, interpolation, name)
    {
    }

    template <typename data_t>
    UpSampling3D<data_t>::UpSampling3D(const std::array<index_t, 3>& size,
                                       Interpolation interpolation, const std::string& name)
        : UpSampling<data_t, LayerType::UpSampling3D, 3>(size, interpolation, name)
    {
    }

    template class Reshape<float>;
    template class Flatten<float>;
    template class UpSampling<float, LayerType::UpSampling2D, 2>;
    template struct UpSampling2D<float>;
} // namespace elsa::ml