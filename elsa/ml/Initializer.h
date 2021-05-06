#pragma once

#include <utility>
#include <optional>
#include <random>

#include "elsaDefines.h"
#include "Common.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            class InitializerImpl
            {
            public:
                using FanPairType = std::pair<index_t, index_t>;

                /// Set a seed for the random generator. If no seed is set
                /// all random generators are initialized using a non-deterministic
                /// random device.
                static void setSeed(uint64_t seed);

                /// Clear the seed for the random generator and use a non-deterministic
                /// random device for initialization.
                static void clearSeed();

                /// Initialize a chunk of raw memory with samples for a given
                /// distribution.
                /// @param data Pointer to memory to be initialized
                /// @param size Size of the memory block to be initialized
                /// @param initializer Specifies how the memory block will be initialized
                /// @param fanInOut Tuple containing the fan-in and fan-out of a layer.
                /// This information is used for some initialization types.
                static void initialize(
                    data_t* data, index_t size, Initializer initializer,
                    [[maybe_unused]] const InitializerImpl<data_t>::FanPairType& fanInOut);

                /// Initialize a chunk of raw memory with samples for a given
                /// distribution.
                /// @param data Pointer to memory to be initialized
                /// @param size Size of the memory block to be initialized
                /// @param initializer Specifies how the memory block will be initialized
                static void initialize(data_t* data, index_t size, Initializer initializer);

            private:
                /// Unform random initialization
                static void uniform(data_t* data, index_t size);

                /// Unform random initialization bounded to an interval
                static void uniform(data_t* data, index_t size, data_t lowerBound,
                                    data_t upperBound);

                /// Initialization with a given constant value
                static void constant(data_t* data, index_t size, data_t value);

                /// Initialization with the constant value 1
                static void ones(data_t* data, index_t size);

                /// Initialization with the constant value 0
                static void zeros(data_t* data, index_t size);

                /**
                 * Normal distribution
                 *
                 * Initialize data with random samples from a normal distribution.
                 */
                static void normal(data_t* data, index_t size, data_t mean, data_t stddev);

                /**
                 * Truncated Normal distribution
                 *
                 * Initialize data with random samples from a normal distribution but discard if
                 * values have a distance of more than 2 x stddev from mean.
                 */
                static void truncatedNormal(data_t* data, index_t size, data_t mean, data_t stddev);

                /**
                 * Glorot uniform initialization
                 *
                 * Initialize data with random samples from a uniform distribution on
                 * the interval
                 *
                 *      `[-sqrt(6 / (fanIn + fanOut)), sqrt(6 / (fanIn + fanOut))]`
                 */
                static void glorotUniform(data_t* data, index_t size, const FanPairType&);

                /**
                 * Glorot normal initialization
                 *
                 * Initialize data  with random samples from a truncated normal distribution
                 * centered on mean 0 and standard deviation
                 *
                 * `sqrt(2 / (fanIn + fanOut))`
                 */
                static void glorotNormal(data_t* data, index_t size, const FanPairType&);

                /**
                 * He normal initialization
                 *
                 * Initialize data  with random samples from a truncated normal distribution
                 * centered on mean 0 and standard deviation
                 *
                 * `sqrt(2 / fanIn)`
                 */
                static void heNormal(data_t* data, index_t size, const FanPairType&);

                /**
                 * Glorot uniform initialization
                 *
                 * Initialize data with random samples from a uniform distribution on
                 * the interval
                 *
                 *      `[-sqrt(6 / fanIn), sqrt(6 / fanIn)]`
                 */
                static void heUniform(data_t* data, index_t size, const FanPairType&);

                static void ramlak(data_t* data, index_t size);

                /// return a Mersenne-Twister engine either initialized by a fixed random seed or by
                /// a random-device
                static std::mt19937_64 getEngine();

                /// Type of the uniform distribution depending on the data-type used
                using UniformDistributionType =
                    std::conditional_t<std::is_integral<data_t>::value,
                                       std::uniform_int_distribution<data_t>,
                                       std::uniform_real_distribution<data_t>>;

                /// Type of the uniform distribution depending on the data-type used
                using NormalDistributionType =
                    std::conditional_t<std::is_floating_point<data_t>::value,
                                       std::normal_distribution<data_t>, std::false_type>;

                /// Optional seed for random distributions
                static uint64_t seed_;
                static bool useSeed_;

                static std::random_device randomDevice_;
            };
        } // namespace detail
    }     // namespace ml
} // namespace elsa