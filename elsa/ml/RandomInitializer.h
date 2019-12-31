#pragma once

#include <random>
#include <type_traits>
#include <optional>
#include "DataContainer.h"

namespace elsa
{
    enum class Initializer {
        Uniform,

        One,

        Zero,

        /**
         * Glorot normal initialization
         *
         * Initialize a data container with a random samples from a truncated
         * normal distribution centered on variance 0 and standard deviation
         *          sqrt(2 / (fan_in + fan_out))
         */
        GlorotNormal,

        /**
         * Glorot uniform initialization
         *
         * Initialize a data container with a random samples from a uniform
         * distribution on the interval
         *      [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
         */
        GlorotUniform,
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
        static void setSeed(const std::optional<data_t>& seed = std::nullopt);

        static void initialize(data_t* data, std::size_t size, Initializer initializer)
        {
            switch (initializer) {
                case Initializer::Uniform:
                    RandomInitializer::uniform(data, size);
                    return;
                case Initializer::One:
                    RandomInitializer::one(data, size);
                    return;
                case Initializer::Zero:
                    RandomInitializer::zero(data, size);
                    return;
                default:
                    throw std::invalid_argument("Unkown random initializer");
            }
        }

    private:
        /// Unform random initialization
        static void uniform(data_t* data, std::size_t size);

        static void uniform(data_t* data, std::size_t size, data_t lowerBound, data_t upperBound);

        /// Initialization with a given constant value
        static void constant(data_t* data, std::size_t size, data_t value);

        static void one(data_t* data, std::size_t size);

        static void zero(data_t* data, std::size_t size);

        /**
         * Glorot normal initialization
         *
         * Initialize a data container with a random samples from a truncated
         * normal distribution centered on variance 0 and standard deviation
         *          sqrt(2 / (fan_in + fan_out))
         */
        // static void glorotNormal(const DataContainer&);

        // /**
        //  * Glorot uniform initialization
        //  *
        //  * Initialize a data container with a random samples from a uniform
        //  * distribution on the interval
        //  *      [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))]
        //  */
        // static void glorotUniform(const DataContainer&);

        // static void heNormal(const DataContainer&);
        // static void heUniform(const DataContainer&);

        using UniformDistributionType =
            std::conditional_t<std::is_integral_v<data_t>, std::uniform_int_distribution<data_t>,
                               std::uniform_real_distribution<data_t>>;

        using NormalDistributionType =
            std::conditional<std::is_floating_point_v<data_t>, std::normal_distribution<data_t>,
                             std::false_type>;

        static std::optional<data_t> _seed;

        static std::random_device _randomDevice;
    };
} // namespace elsa
