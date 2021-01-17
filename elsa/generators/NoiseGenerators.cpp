#include "NoiseGenerators.h"

namespace elsa
{
    template <typename data_t>
    DataContainer<data_t> NoNoiseGenerator::operator()(const DataContainer<data_t>& dc) const
    {
        return dc;
    }

    template <typename data_t>
    DataContainer<data_t> GaussianNoiseGenerator::operator()(const DataContainer<data_t>& dc) const
    {
        // Define random generator with Gaussian distribution
        std::default_random_engine generator;
        std::normal_distribution<data_t> dist(_mean, _stddev);

        auto newDc = dc;

        for (int i = 0; i < dc.getSize(); ++i) {
            newDc[i] += dist(generator);
        }
        return newDc;
    }

    template <typename data_t>
    DataContainer<data_t> PoissonNoiseGenerator::operator()(const DataContainer<data_t>& dc) const
    {
        // Define random generator with Gaussian distribution
        std::default_random_engine generator;
        std::exponential_distribution<data_t> dist(_mean);

        auto newDc = dc;

        for (int i = 0; i < dc.getSize(); ++i) {
            newDc[i] += dist(generator);
        }
        return newDc;
    }

    // Explicit template instantiations
    template DataContainer<float>
        NoNoiseGenerator::operator()<float>(const DataContainer<float>&) const;
    template DataContainer<double>
        NoNoiseGenerator::operator()<double>(const DataContainer<double>&) const;

    template DataContainer<float>
        GaussianNoiseGenerator::operator()<float>(const DataContainer<float>&) const;
    template DataContainer<double>
        GaussianNoiseGenerator::operator()<double>(const DataContainer<double>&) const;

    template DataContainer<float>
        PoissonNoiseGenerator::operator()<float>(const DataContainer<float>&) const;
    template DataContainer<double>
        PoissonNoiseGenerator::operator()<double>(const DataContainer<double>&) const;

} // namespace elsa
