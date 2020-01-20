
#include "RandomInitializer.h"

namespace elsa
{
    template <typename data_t>
    std::random_device RandomInitializer<data_t>::_randomDevice{};

    template <typename data_t>
    std::optional<uint64_t> RandomInitializer<data_t>::_seed = std::nullopt;

    template <typename data_t>
    void RandomInitializer<data_t>::setSeed(uint64_t seed)
    {
        _seed = std::optional<uint64_t>(seed);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::clearSeed()
    {
        _seed = std::nullopt;
    }

    template <typename data_t>
    void RandomInitializer<data_t>::uniform(data_t* data, index_t size, data_t lowerBound,
                                            data_t upperBound)
    {
        UniformDistributionType dist(lowerBound, upperBound);
        std::mt19937_64 engine;

        if (_seed.has_value())
            engine = std::mt19937_64(_seed.value());
        else
            engine = std::mt19937_64(_randomDevice());

        for (index_t i = 0; i < size; ++i)
            data[i] = dist(engine);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::uniform(data_t* data, index_t size)
    {
        RandomInitializer<data_t>::uniform(data, size, 0, std::numeric_limits<data_t>::max());
    }

    template <typename data_t>
    void RandomInitializer<data_t>::constant(data_t* data, index_t size, data_t constant)
    {
        for (index_t i = 0; i < size; ++i)
            data[i] = constant;
    }

    template <typename data_t>
    void RandomInitializer<data_t>::one(data_t* data, index_t size)
    {
        constant(data, size, static_cast<data_t>(1));
    }

    template <typename data_t>
    void RandomInitializer<data_t>::zero(data_t* data, index_t size)
    {
        constant(data, size, static_cast<data_t>(0));
    }

    template <typename data_t>
    void RandomInitializer<data_t>::glorotUniform(data_t* data, index_t size,
                                                  const FanPairType& fan)
    {
        auto bound = static_cast<data_t>(
            std::sqrt(6 / (static_cast<data_t>(fan.first) + static_cast<data_t>(fan.second))));
        uniform(data, size, -1 * bound, bound);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::glorotNormal(data_t* data, index_t size, const FanPairType& fan)
    {
        auto stddev = static_cast<data_t>(
            std::sqrt(2 / (static_cast<data_t>(fan.first) + static_cast<data_t>(fan.second))));
        truncatedNormal(data, size, 0, stddev);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::normal(data_t* data, index_t size, data_t mean, data_t stddev)
    {
        if constexpr (std::is_same_v<std::false_type, NormalDistributionType>) {
            throw std::logic_error("Cannot use normal distribution with the given data-type");
        } else {
            NormalDistributionType dist(mean, stddev);
            std::mt19937_64 engine;

            if (_seed.has_value())
                engine = std::mt19937_64(_seed.value());
            else
                engine = std::mt19937_64(_randomDevice());

            for (index_t i = 0; i < size; ++i)
                data[i] = dist(engine);
        }
    }

    template <typename data_t>
    void RandomInitializer<data_t>::truncatedNormal(data_t* data, index_t size, data_t mean,
                                                    data_t stddev)
    {
        if constexpr (std::is_same_v<std::false_type, NormalDistributionType>) {
            throw std::logic_error("Cannot use normal distribution with the given data-type");
        } else {
            NormalDistributionType dist(mean, stddev);
            std::mt19937_64 engine;

            if (_seed.has_value())
                engine = std::mt19937_64(_seed.value());
            else
                engine = std::mt19937_64(_randomDevice());

            for (index_t i = 0; i < size; ++i) {
                auto value = dist(engine);
                while (std::abs(mean - value) > 2 * stddev) {
                    value = dist(engine);
                }
                data[i] = value;
            }
        }
    }

    template <typename data_t>
    void RandomInitializer<data_t>::heNormal(data_t* data, index_t size,
                                             const FanPairType& fanInOut)
    {
        auto stddev = std::sqrt(static_cast<data_t>(2) / fanInOut.first);
        truncatedNormal(data, size, 0, stddev);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::heUniform(data_t* data, index_t size, const FanPairType& fan)
    {
        auto bound = static_cast<data_t>(std::sqrt(6 / (static_cast<data_t>(fan.first))));
        uniform(data, size, -1 * bound, bound);
    }

    template class RandomInitializer<float>;
} // namespace elsa