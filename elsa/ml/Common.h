#pragma once

#include <string>

#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

namespace elsa
{
    namespace ml
    {
        /// Initializer that can be used to initialize trainable parameters in a network layer
        enum class Initializer {
            /**
             * Ones initialization
             *
             * Initialize data with \f$ 1 \f$
             */
            Ones,

            /**
             * Zeros initialization
             *
             * Initialize data with \f$ 0 \f$
             */
            Zeros,

            /**
             * Uniform initialization
             *
             * Initialize data with random samples from a uniform distribution in
             * the interval \f$ [-1, 1 ] \f$.
             */
            Uniform,

            /**
             * Normal initialization
             *
             * Initialize data with random samples from a standard normal
             * distribution, i.e., a normal distribution with mean 0 and
             * standard deviation \f$ 1 \f$.
             */
            Normal,

            /**
             * Truncated normal initialization
             *
             * Initialize data with random samples from a truncated standard normal
             * distribution, i.e., a normal distribution with mean 0 and standard deviation \f$ 1
             * \f$ where values with a distance of greater than \f$ 2 \times \f$ standard deviations
             * from the mean are discarded.
             */
            TruncatedNormal,

            /**
             * Glorot uniform initialization
             *
             * Initialize a data container with a random samples from a uniform
             * distribution on the interval
             * \f$ \left [ - \sqrt{\frac{6}{\text{fanIn} + \text{fanOut}}} ,
             * \sqrt{\frac{6}{\text{fanIn} + \text{fanOut}}} \right ] \f$
             */
            GlorotUniform,

            /**
             * Glorot normal initialization
             *
             * Initialize data with random samples from a truncated normal distribution
             * with mean \f$ 0 \f$ and stddev \f$ \sqrt{ \frac{2}{\text{fanIn} + \text{fanOut}}}
             * \f$.
             */
            GlorotNormal,

            /**
             * He normal initialization
             *
             * Initialize data with random samples from a truncated normal distribution
             * with mean \f$ 0 \f$ and stddev \f$ \sqrt{\frac{2}{\text{fanIn}}} \f$
             */
            HeNormal,

            /**
             * He uniform initialization
             *
             * Initialize a data container with a random samples from a uniform
             * distribution on the interval \f$  \left [ - \sqrt{\frac{6}{\text{fanIn}}} ,
             * \sqrt{\frac{6}{\text{fanIn}}} \right ] \f$
             */
            HeUniform,

            /**
             * RamLak filter initialization
             *
             * Initialize data with values of the RamLak filter, the discrete
             * version of the Ramp filter in the spatial domain.
             *
             * Values for this initialization are given by the following
             * equation:
             *
             * \f[
             *  \text{data}[i] = \begin{cases}
             *  \frac{1}{i^2 \pi^2}, & i \text{ even} \\
             *  \frac 14, & i = \frac{\text{size}-1}{2} \\
             *  0, & i \text{ odd}.
             * \end{cases}
             * \f]
             */
            RamLak
        };

        /// Padding type for Pooling and Convolutional layers
        enum class Padding {
            /// Do not pad the input
            Valid,
            /// Pad the input such that the output shape matches the input shape.
            Same
        };

        /// Backend to execute model primitives.
        enum class MlBackend {
            /// Automatically choose the fastest backend available.
            Auto,
            /// Use the Dnnl, aka. OneDNN backend which is optimized for CPUs.
            Dnnl,
            /// Use the Cudnn backend which is optimized for Nvidia GPUs.
            Cudnn
        };

        /// Type of the interpolation for Upsampling
        enum class Interpolation {
            /// Perform nearest neighbour interpolarion
            NearestNeighbour,
            /// Perform bilinear interpolarion
            Bilinear
        };

        /// type of a network layer
        enum class LayerType {
            Undefined,
            Input,
            Dense,
            Activation,
            Sigmoid,
            Relu,
            Tanh,
            ClippedRelu,
            Elu,
            Identity,
            Conv1D,
            Conv2D,
            Conv3D,
            Conv2DTranspose,
            Conv3DTranspose,
            MaxPooling1D,
            MaxPooling2D,
            MaxPooling3D,
            AveragePooling1D,
            AveragePooling2D,
            AveragePooling3D,
            Sum,
            Concatenate,
            Reshape,
            Flatten,
            Softmax,
            UpSampling1D,
            UpSampling2D,
            UpSampling3D,
            Projector
        };

        /// Direction of data propagation through a network
        enum class PropagationKind {
            /// perform a forward propagation
            Forward,
            /// perform a backward propagation
            Backward,
            // perform both, forward and backward propagation
            Full
        };

        /// Activation function for Dense and Convolutional layers
        enum class Activation {
            /// Sigmoid activation function
            Sigmoid,
            /// Relu activation function
            Relu,
            /// Clipped Relu activation function.
            ClippedRelu,
            /// Hyperbolic tangent activation function.
            Tanh,
            /// Exponential Linear Unit.
            Elu,
            /// Identity activation
            Identity
        };

        namespace detail
        {
            std::string getEnumMemberAsString(LayerType);
            std::string getEnumMemberAsString(Initializer);
            std::string getEnumMemberAsString(MlBackend);
            std::string getEnumMemberAsString(PropagationKind);
        } // namespace detail
    }     // namespace ml
} // namespace elsa
