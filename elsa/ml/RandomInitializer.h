#pragma once

#include <random>
#include <type_traits>
#include <optional>
#include "DataContainer.h"

namespace elsa
{
    enum class Initializer {
        /**
         * One initialization
         *
         * Initialize data with 1
         */
        One,

        /**
         * Zero initialization
         *
         * Initialize data with 0
         */
        Zero,

        /**
         * Uniform initialization
         *
         * Initialize data with random samples from a uniform distribution in
         * the interval [-1, 1].
         */
        Uniform,

        /**
         * Normal initialization
         *
         * Initialize data with random samples from a standard normal
         * distribution, i.e., a normal distribution with mean 0 and
         * standard deviation 1.
         */
        Normal,

        /**
         * Glorot uniform initialization
         *
         * Initialize a data container with a random samples from a uniform
         * distribution on the interval
         *
         *      `[-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]`
         */
        GlorotUniform,

        /**
         * Glorot normal initialization
         *
         * Initialize data  with random samples from a normal distribution
         * centered on variance 0 and standard deviation
         *
         * `sqrt(2 / (fanIn + fanOut))`
         */
        GlorotNormal,

        HeNormal,
        HeUniform,

        LeCunNormal,
        LeCunUniform
    };

    /**
     * Struct that provides functionality for proper random initialization of
     * trainable parameters in a neural network layer
     */
    template <typename data_t>
    class RandomInitializer final
    {
    public:
        using FanPairType = std::pair<index_t, index_t>;

        RandomInitializer() = default;

        static void setSeed(uint64_t seed);

        static void clearSeed();

        static void initialize(data_t* data, index_t size, Initializer initializer,
                               [[maybe_unused]] const FanPairType& fanInOut)
        {
            switch (initializer) {
                case Initializer::One:
                    RandomInitializer::one(data, size);
                    return;
                case Initializer::Zero:
                    RandomInitializer::zero(data, size);
                    return;
                case Initializer::Uniform:
                    RandomInitializer::uniform(data, size, -1, 1);
                    return;
                case Initializer::GlorotUniform:
                    RandomInitializer::glorotUniform(data, size, fanInOut);
                    return;
                case Initializer::Normal:
                    RandomInitializer::normal(data, size, 0, 1);
                case Initializer::GlorotNormal:
                    RandomInitializer::glorotNormal(data, size, fanInOut);
                    return;
                default:
                    throw std::invalid_argument("Unkown random initializer");
            }
        }

    protected:
        /// Unform random initialization
        static void uniform(data_t* data, index_t size);

        /// Unform random initialization bounded to an interval
        static void uniform(data_t* data, index_t size, data_t lowerBound, data_t upperBound);

        /// Initialization with a given constant value
        static void constant(data_t* data, index_t size, data_t value);

        static void one(data_t* data, index_t size);

        static void zero(data_t* data, index_t size);

        /**
         * Normal distribution
         *
         * Initialize data with random samples from a normal distribution.
         */
        static void normal(data_t* data, index_t size, data_t variance, data_t stddev);

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
         * Initialize data  with random samples from a normal distribution
         * centered on mean 0 and standard deviation
         *
         * `sqrt(2 / (fanIn + fanOut))`
         */
        static void glorotNormal(data_t* data, index_t size, const FanPairType&);

        /// Type of the uniform distribution depending on the data-type used
        using UniformDistributionType =
            std::conditional_t<std::is_integral_v<data_t>, std::uniform_int_distribution<data_t>,
                               std::uniform_real_distribution<data_t>>;

        /// Type of the uniform distribution depending on the data-type used
        using NormalDistributionType =
            std::conditional_t<std::is_floating_point_v<data_t>, std::normal_distribution<data_t>,
                               std::false_type>;

        static std::optional<uint64_t> _seed;

        static std::random_device _randomDevice;
    };
} // namespace elsa
