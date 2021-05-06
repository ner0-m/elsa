#pragma once

#include <memory>
#include <string>
#include <vector>

#include "elsaDefines.h"
#include "Common.h"
#include "Trainable.h"

namespace elsa::ml
{
    /// Base class for all convolutional and transposed-convolutional layers
    template <typename data_t>
    class ConvBase : public Trainable<data_t>
    {
    public:
        /// return the number of filters
        index_t getNumberOfFilters() const;

        /// return the filter data descriptor
        VolumeDescriptor getFilterDescriptor() const;

        /// return the strides
        index_t getStrides() const;

        /// return the padding
        Padding getPadding() const;

    protected:
        /// Construct a ConvBase instance by definining
        /// @param layerType the type of the conv layer
        /// @param activation the activation of the conv layer
        /// @param useBias true of the layer uses a bias, false otherwise
        /// @param kernelInitializer the initializer used to initialize all
        ///                          kernels of the conv layer
        /// @param biasInitializer the initializer use to initialize the bias
        ///                        term. This has no effect if useBias is set to
        ///                        false
        /// @param name the name of this layer
        /// @param requiredNumberOfDimensions the number of dimensions any input
        ///                                   to the layer must have
        ConvBase(LayerType layerType, index_t numberOfFilters,
                 const VolumeDescriptor& filterDescriptor, Activation activation, index_t strides,
                 Padding padding, bool useBias, Initializer kernelInitializer,
                 Initializer biasInitializer, const std::string& name,
                 int requiredNumberOfDimensions);

        index_t numberOfFilters_;
        std::unique_ptr<DataDescriptor> filterDescriptor_;
        Activation activation_;
        index_t strides_;
        Padding padding_;
    };

    template <typename data_t>
    class Conv : public ConvBase<data_t>
    {
    public:
        void computeOutputDescriptor() override;

        /// @return an IndexVector_t with the padding along each dimension
        IndexVector_t getPaddingSizes() const;

    protected:
        Conv(index_t convolutionDimensions, index_t numberOfFilters,
             const VolumeDescriptor& filterDescriptor, Activation activation, index_t strides,
             Padding padding, bool useBias, Initializer kernelInitializer,
             Initializer biasInitializer, const std::string& name);

        index_t convolutionDimensions_;
    };

    /// 1D convolution layer (e.g. temporal convolution).
    ///
    /// @author David Tellenbach
    ///
    /// This layer creates a convolution kernel that is convolved with the layer input over a single
    /// spatial (or temporal) dimension.  If ``useBias`` is ``true``,
    /// a bias vector is added to the outputs.
    template <typename data_t = real_t>
    struct Conv1D : public Conv<data_t> {
        /// Construct a Conv1D layer
        ///
        /// @param numberOfFilters The number of filters for the convolution
        /// @param filterDescriptor A VolumeDescriptor describing the shape of
        /// all filters.
        /// @param activation The activation function finally applied to the
        /// outputs.
        /// @param strides The strides for the convolution. This parameter is
        /// optional and defaults to 1.
        /// @param padding The input padding that is applied before the convolution.
        /// This parameter is optional and defaults to Padding::Valid.
        /// @param useBias True if the layer uses a bias vector, false otherwise.
        /// This parameter is optional and defaults to ``true``.
        /// @param kernelInitializer The initializer used for all convolutional
        /// filters. This parameter is optional and defaults to Initializer::GlorotUniform.
        /// @param biasInitializer The initializer used to initialize the bias vector.
        /// If ``useBias`` is ``false`` this has no effect. This parameter is optional
        /// and defaults to Initializer::Zeros.
        /// @param name The name of this layer.
        Conv1D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
               Activation activation, index_t strides = 1, Padding padding = Padding::Valid,
               bool useBias = true, Initializer kernelInitializer = Initializer::GlorotUniform,
               Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");

        Conv1D(index_t numberOfFilters, const std::array<index_t, 2>& filterSize,
               Activation activation, index_t strides = 1, Padding padding = Padding::Valid,
               bool useBias = true, Initializer kernelInitializer = Initializer::GlorotUniform,
               Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");
    };

    /// 2D convolution layer (e.g. spatial convolution over images).
    ///
    /// @author David Tellenbach
    ///
    /// This layer implements a spatial convolution layer with a given number of filters that is
    /// convolved over the spatial dimensions of an image. If ``useBias`` is ``true``, a bias vector
    /// is added to the outputs.
    template <typename data_t = real_t>
    struct Conv2D : public Conv<data_t> {

        /// Construct a Conv2D layer
        ///
        /// @param numberOfFilters The number of filters for the convolution
        /// @param filterDescriptor A VolumeDescriptor describing the shape of
        /// all filters.
        /// @param activation The activation function finally applied to the
        /// outputs.
        /// @param strides The strides for the convolution. This parameter is
        /// optional and defaults to 1.
        /// @param padding The input padding that is applied before the convolution.
        /// This parameter is optional and defaults to Padding::Valid.
        /// @param useBias True if the layer uses a bias vector, false otherwise.
        /// This parameter is optional and defaults to ``true``.
        /// @param kernelInitializer The initializer used for all convolutional
        /// filters. This parameter is optional and defaults to Initializer::GlorotUniform.
        /// @param biasInitializer The initializer used to initialize the bias vector.
        /// If ``useBias`` is ``false`` this has no effect. This parameter is optional
        /// and defaults to Initializer::Zeros.
        /// @param name The name of this layer.
        Conv2D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
               Activation activation, index_t strides = 1, Padding padding = Padding::Valid,
               bool useBias = true, Initializer kernelInitializer = Initializer::GlorotUniform,
               Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");

        /// Construct a Conv2D layer
        ///
        /// @param numberOfFilters The number of filters for the convolution
        /// @param filterSize The size of all filters as an ``std::array``, e.g. ``{w, h, c}``.
        /// @param activation The activation function finally applied to the outputs.
        /// @param strides The strides for the convolution. This parameter is optional and defaults
        /// to 1. @param padding The input padding that is applied before the convolution. This
        /// parameter is optional and defaults to Padding::Valid.
        /// @param useBias True if the layer
        /// uses a bias vector, false otherwise. This parameter is optional and defaults to
        /// ``true``.
        /// @param kernelInitializer The initializer used for all convolutional filters.
        /// This parameter is optional and defaults to Initializer::GlorotUniform.
        /// @param biasInitializer The initializer used to initialize the bias vector. If
        /// ``useBias`` is
        /// ``false`` this has no effect. This parameter is optional and defaults to
        /// Initializer::Zeros.
        /// @param name The name of this layer.
        Conv2D(index_t numberOfFilters, const std::array<index_t, 3>& filterSize,
               Activation activation, index_t strides = 1, Padding padding = Padding::Valid,
               bool useBias = true, Initializer kernelInitializer = Initializer::GlorotUniform,
               Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");
    };

    /// 3D convolution layer (e.g. spatial convolution over volumes).
    ///
    /// @author David Tellenbach
    ///
    /// This layer implements a spatial convolution layer with a given number of filters that is
    /// convolved over the spatial dimensions of a volume. If ``useBias`` is ``true``, a bias vector
    /// is added to the outputs.
    template <typename data_t = real_t>
    struct Conv3D : public Conv<data_t> {
        /// Construct a Conv3D layer
        ///
        /// @param numberOfFilters The number of filters for the convolution
        /// @param filterDescriptor A VolumeDescriptor describing the shape of
        /// all filters.
        /// @param activation The activation function finally applied to the
        /// outputs.
        /// @param strides The strides for the convolution. This parameter is
        /// optional and defaults to 1.
        /// @param padding The input padding that is applied before the convolution.
        /// This parameter is optional and defaults to Padding::Valid.
        /// @param useBias True if the layer uses a bias vector, false otherwise.
        /// This parameter is optional and defaults to ``true``.
        /// @param kernelInitializer The initializer used for all convolutional
        /// filters. This parameter is optional and defaults to Initializer::GlorotUniform.
        /// @param biasInitializer The initializer used to initialize the bias vector.
        /// If ``useBias`` is ``false`` this has no effect. This parameter is optional
        /// and defaults to Initializer::Zeros.
        /// @param name The name of this layer.
        Conv3D(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
               Activation activation, index_t strides = 1, Padding padding = Padding::Valid,
               bool useBias = true, Initializer kernelInitializer = Initializer::GlorotUniform,
               Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");
    };

    template <typename data_t>
    class ConvTranspose : public ConvBase<data_t>
    {
    public:
        /// Calculate the layer's output descriptor using the following formula:
        ///
        /// Given
        ///   i,   the input-size,
        ///   p,   the padding
        ///   s,   the stride
        ///   k,   the kernel-size,
        /// the output-size is given by
        ///     o = s(i-1)+k-2p
        void computeOutputDescriptor() override;

    protected:
        ConvTranspose(index_t convolutionDimensions, index_t numberOfFilters,
                      const VolumeDescriptor& filterDescriptor, Activation activation,
                      index_t strides, Padding padding, bool useBias, Initializer kernelInitializer,
                      Initializer biasInitializer, const std::string& name);

        index_t convolutionDimensions_;
    };

    /// Transposed convolution layer (sometimes called Deconvolution).
    ///
    /// @author David Tellenbach
    ///
    /// The need for transposed convolutions generally arises from the desire to use a
    /// transformation going in the opposite direction of a normal convolution, i.e., from something
    /// that has the shape of the output of some convolution to something that has the shape of its
    /// input while maintaining a connectivity pattern that is compatible with said convolution.
    template <typename data_t = real_t>
    struct Conv2DTranspose : public ConvTranspose<data_t> {
        /// Construct a Conv2DTranspose layer
        ///
        /// @param numberOfFilters The number of filters for the convolution
        /// @param filterDescriptor A VolumeDescriptor describing the shape of
        /// all filters.
        /// @param activation The activation function finally applied to the
        /// outputs.
        /// @param strides The strides for the convolution. This parameter is
        /// optional and defaults to 1.
        /// @param padding The input padding that is applied before the convolution.
        /// This parameter is optional and defaults to Padding::Valid.
        /// @param useBias True if the layer uses a bias vector, false otherwise.
        /// This parameter is optional and defaults to ``true``.
        /// @param kernelInitializer The initializer used for all convolutional
        /// filters. This parameter is optional and defaults to Initializer::GlorotUniform.
        /// @param biasInitializer The initializer used to initialize the bias vector.
        /// If ``useBias`` is ``false`` this has no effect. This parameter is optional
        /// and defaults to Initializer::Zeros.
        /// @param name The name of this layer.
        Conv2DTranspose(index_t numberOfFilters, const VolumeDescriptor& filterDescriptor,
                        Activation activation, index_t strides = 1,
                        Padding padding = Padding::Valid, bool useBias = true,
                        Initializer kernelInitializer = Initializer::GlorotUniform,
                        Initializer biasInitializer = Initializer::Zeros,
                        const std::string& name = "");
    };

} // namespace elsa::ml