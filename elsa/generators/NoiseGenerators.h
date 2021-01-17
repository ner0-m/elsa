#pragma once

#include "DataContainer.h"
#include <random>

namespace elsa
{

    struct NoNoiseGenerator {
    public:
        template <typename data_t>
        DataContainer<data_t> operator()(const DataContainer<data_t>& dc) const;
    };

    struct GaussianNoiseGenerator {
    public:
        GaussianNoiseGenerator(real_t mean, real_t stddev) : _mean(mean), _stddev(stddev) {}

        template <typename data_t>
        DataContainer<data_t> operator()(const DataContainer<data_t>& dc) const;

    private:
        real_t _mean{0.0};
        real_t _stddev{0.0};
    };

    struct PoissonNoiseGenerator {
    public:
        PoissonNoiseGenerator(real_t mean) : _mean(mean) {}

        template <typename data_t>
        DataContainer<data_t> operator()(const DataContainer<data_t>& dc) const;

    private:
        real_t _mean{0.0};
    };
} // namespace elsa
