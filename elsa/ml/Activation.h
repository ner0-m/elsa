#pragma once

#include <string>

#include "elsaDefines.h"
#include "VolumeDescriptor.h"
#include "Layer.h"
#include "Common.h"

namespace elsa::ml
{
    template <typename data_t = real_t>
    class ActivationBase : public Layer<data_t>
    {
    public:
        /// Construct an Activation layer
        ///
        /// \param activation
        /// \param name
        explicit ActivationBase(Activation activation, const std::string& name = "");

        /// default constructor
        ActivationBase() = default;

        /// \returns the activation
        Activation getActivation() const;

    private:
        Activation activation_;
    };

    /**
     * \brief Sigmoid activation function, \f$ \text{sigmoid}(x) = 1 / (1 + \exp(-x)) \f$.
     *
     * \author David Tellenbach
     */
    template <typename data_t = real_t>
    struct Sigmoid : ActivationBase<data_t> {
        explicit Sigmoid(const std::string& name = "");
    };

    /**
     * \brief Applies the rectified linear unit activation function.
     *
     * \author David Tellenbach
     *
     * With default values, this returns the standard ReLU activation \f$ \max(x, 0) \f$,
     * the element-wise maximum of 0 and the input tensor.
     */
    template <typename data_t = real_t>
    struct Relu : ActivationBase<data_t> {
        explicit Relu(const std::string& name = "");
    };

    /// \brief Hyperbolic tangent activation function.
    ///
    /// \author David Tellenbach
    template <typename data_t = real_t>
    struct Tanh : ActivationBase<data_t> {
        explicit Tanh(const std::string& name = "");
    };

    /// \brief Clipped Relu activation function.
    ///
    /// \author David Tellenbach
    template <typename data_t = real_t>
    struct ClippedRelu : ActivationBase<data_t> {
        explicit ClippedRelu(const std::string& name = "");
    };

    /// \brief Exponential Linear Unit.
    ///
    /// \author David Tellenbach
    ///
    /// ELUs have negative values which pushes the mean of the activations
    /// closer to zero. Mean activations that are closer to zero enable faster
    /// learning as they bring the gradient closer to the natural gradient. ELUs
    /// saturate to a negative value when the argument gets smaller. Saturation
    /// means a small derivative which decreases the variation and the
    /// information that is propagated to the next layer.
    template <typename data_t = real_t>
    struct Elu : ActivationBase<data_t> {
        explicit Elu(const std::string& name = "");
    };

    /// \brief Identity activation
    ///
    /// \author David Tellenbach
    ///
    /// This is not a real activation function and just leaves the input
    /// unchanged.
    template <typename data_t = real_t>
    struct IdentityActivation : ActivationBase<data_t> {
        explicit IdentityActivation(const std::string& name = "");
    };

} // namespace elsa::ml