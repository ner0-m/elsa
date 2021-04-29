#pragma once

#include <string>

#include "Common.h"
#include "Layer.h"

namespace elsa::ml
{
    /// A reshape layer.
    ///
    /// \author David Tellenbach
    ///
    /// Reshapes the input while leaving the data unchanged.
    template <typename data_t = real_t>
    class Reshape : public Layer<data_t>
    {
    public:
        /// Construct a Reshape layer by specifying the target-shape.
        explicit Reshape(const VolumeDescriptor& targetShape, const std::string& name = "");

        /// \copydoc Layer::computeOutputDescriptor
        void computeOutputDescriptor() override;
    };

    /// A flatten layer.
    ///
    /// \author David Tellenbach
    ///
    /// Flattens the input while leaving the data unchanged.
    template <typename data_t = real_t>
    class Flatten : public Layer<data_t>
    {
    public:
        /// Construction a Flatten layer.
        explicit Flatten(const std::string& name = "");

        /// \copydoc Layer::computeOutputDescriptor
        void computeOutputDescriptor() override;
    };

    template <typename data_t, LayerType layerType, index_t UpSamplingDimensions>
    class UpSampling : public Layer<data_t>
    {
    public:
        explicit UpSampling(const std::array<index_t, UpSamplingDimensions>& size,
                            Interpolation interpolation, const std::string& name = "");

        void computeOutputDescriptor() override;

        Interpolation getInterpolation() const;

    private:
        std::array<index_t, UpSamplingDimensions> size_;
        Interpolation interpolation_;
    };

    /// Upsampling layer for 1D inputs.
    ///
    /// \author David Tellenbach
    ///
    /// Repeats each temporal step size times along the time axis.
    template <typename data_t = real_t>
    struct UpSampling1D : public UpSampling<data_t, LayerType::UpSampling1D, 1> {
        /// Construct an UpSampling1D layer
        ///
        /// \param size The upsampling factors for dim1.
        /// \param interpolation The interpolation used to upsample the input.
        /// This parameter is optional and defaults to Interpolation::NearestNeighbour.
        /// \param name The name of this layer.
        explicit UpSampling1D(const std::array<index_t, 1>& size,
                              Interpolation interpolation = Interpolation::NearestNeighbour,
                              const std::string& name = "");
    };

    /// Upsampling layer for 2D inputs.
    ///
    /// \author David Tellenbach
    ///
    /// Repeats the rows and columns of the data by size[0] and size[1] respectively.
    template <typename data_t = real_t>
    struct UpSampling2D : public UpSampling<data_t, LayerType::UpSampling2D, 2> {
        /// Construct an UpSampling2D layer
        ///
        /// \param size The upsampling factors for dim1 and dim2.
        /// \param interpolation The interpolation used to upsample the input.
        /// This parameter is optional and defaults to Interpolation::NearestNeighbour.
        /// \param name The name of this layer.
        explicit UpSampling2D(const std::array<index_t, 2>& size,
                              Interpolation interpolation = Interpolation::NearestNeighbour,
                              const std::string& name = "");
    };

    /// Upsampling layer for 3D inputs.
    ///
    /// \author David Tellenbach
    ///
    /// Repeats the 1st, 2nd and 3rd dimensions of the data by size[0], size[1]
    /// and size[2] respectively.
    template <typename data_t = real_t>
    struct UpSampling3D : public UpSampling<data_t, LayerType::UpSampling3D, 3> {
        /// Construct an UpSampling3D layer
        ///
        /// \param size The upsampling factors for dim1, dim2 and dim3.
        /// \param interpolation The interpolation used to upsample the input.
        /// This parameter is optional and defaults to Interpolation::NearestNeighbour.
        /// \param name The name of this layer.
        explicit UpSampling3D(const std::array<index_t, 3>& size,
                              Interpolation interpolation = Interpolation::NearestNeighbour,
                              const std::string& name = "");
    };
} // namespace elsa::ml