
#include "RandomInitializer.h"
#include "Logger.h"

namespace elsa
{
    template <typename data_t>
    std::random_device RandomInitializer<data_t>::_randomDevice{};

    template <typename data_t>
    std::optional<data_t> RandomInitializer<data_t>::_seed = std::nullopt;

    template <typename data_t>
    void RandomInitializer<data_t>::setSeed(const std::optional<data_t>& seed)
    {
        _seed = seed;
    }

    template <typename data_t>
    void RandomInitializer<data_t>::uniform(data_t* data, std::size_t size, data_t lowerBound,
                                            data_t upperBound)
    {
        UniformDistributionType dist(lowerBound, upperBound);
        std::mt19937_64 engine;

        if (_seed.value_or(false))
            engine = std::mt19937_64(_seed.value());
        else
            engine = std::mt19937_64(_randomDevice());

        for (std::size_t i = 0; i < size; ++i)
            data[i] = dist(engine);
    }

    template <typename data_t>
    void RandomInitializer<data_t>::uniform(data_t* data, std::size_t size)
    {
        RandomInitializer<data_t>::uniform(data, size, 0, std::numeric_limits<data_t>::max());
    }

    template <typename data_t>
    void RandomInitializer<data_t>::constant(data_t* data, std::size_t size, data_t constant)
    {
        for (std::size_t i = 0; i < size; ++i)
            data[i] = constant;
    }

    template <typename data_t>
    void RandomInitializer<data_t>::one(data_t* data, std::size_t size)
    {
        constant(data, size, static_cast<data_t>(1));
    }

    template <typename data_t>
    void RandomInitializer<data_t>::zero(data_t* data, std::size_t size)
    {
        constant(data, size, static_cast<data_t>(0));
    }

    template class RandomInitializer<float>;
} // namespace elsa