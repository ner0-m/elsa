#pragma once

#include <string>

#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "Trainable.h"
#include "Common.h"

namespace elsa::ml
{
    /// \brief Just your regular densely-connected NN layer.
    ///
    /// \author David Tellenbach
    ///
    /// Dense implements the operation:
    ///   \f$ \text{output} = \text{activation}(\text{input} \cdot \text{kernel} + \text{bias}) \f$
    /// where activation is the element-wise activation function passed as the
    /// activation argument, kernel is a weights matrix created by the layer,
    /// and bias is a bias vector created by the layer (only applicable if
    /// ``use_bias`` is ``true``).
    ///
    /// \author David Tellenbach
    template <typename data_t = real_t>
    class Dense : public Trainable<data_t>
    {
    public:
        /// Construct a Dense layer
        ///
        /// \param units The number of units (neurons) of the layer. This is
        /// also the dimensionality of the output space.
        /// \param activation Activation function to use.
        /// \param useBias Whether the layer uses a bias vector.
        /// \param kernelInitializer Initializer for the kernel weights matrix.
        /// This parameter is optional and defaults to Initializer::GlorotNormal.
        /// \param biasInitializer Initializer for the bias vector. This
        /// parameter is optional and defaults to Initializer::Zeros.
        /// \param name
        Dense(index_t units, Activation activation, bool useBias = true,
              Initializer kernelInitializer = Initializer::GlorotNormal,
              Initializer biasInitializer = Initializer::Zeros, const std::string& name = "");

        /// Default constructor
        Dense() = default;

        /// \returns the the number of units (also called neurons) of this layer.
        index_t getNumberOfUnits() const;

        /// \copydoc Trainable::computeOutputDescriptor
        void computeOutputDescriptor() override;

    private:
        index_t units_;
    };

} // namespace elsa::ml
