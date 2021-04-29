#include "Initializer.h"

namespace elsa
{
    namespace ml
    {
        namespace detail
        {
            template <typename data_t>
            std::random_device InitializerImpl<data_t>::randomDevice_{};

            template <typename data_t>
            bool InitializerImpl<data_t>::useSeed_ = false;

            template <typename data_t>
            uint64_t InitializerImpl<data_t>::seed_ = 1;

            template <typename data_t>
            void InitializerImpl<data_t>::setSeed(uint64_t seed)
            {
                seed_ = seed;
                useSeed_ = true;
            }

            template <typename data_t>
            void InitializerImpl<data_t>::clearSeed()
            {
                useSeed_ = false;
            }

            template <typename data_t>
            void InitializerImpl<data_t>::initialize(
                data_t* data, index_t size, Initializer initializer,
                [[maybe_unused]] const InitializerImpl<data_t>::FanPairType& fanInOut)
            {
                switch (initializer) {
                    case Initializer::Ones:
                        InitializerImpl::ones(data, size);
                        return;
                    case Initializer::Zeros:
                        InitializerImpl::zeros(data, size);
                        return;
                    case Initializer::Uniform:
                        InitializerImpl::uniform(data, size, -1, 1);
                        return;
                    case Initializer::GlorotUniform:
                        InitializerImpl::glorotUniform(data, size, fanInOut);
                        return;
                    case Initializer::HeUniform:
                        InitializerImpl::heUniform(data, size, fanInOut);
                        return;
                    case Initializer::TruncatedNormal:
                        InitializerImpl::truncatedNormal(data, size, 0, 1);
                        return;
                    case Initializer::Normal:
                        InitializerImpl::normal(data, size, 0, 1);
                        return;
                    case Initializer::GlorotNormal:
                        InitializerImpl::glorotNormal(data, size, fanInOut);
                        return;
                    case Initializer::RamLak:
                        InitializerImpl::ramlak(data, size);
                        return;
                    default:
                        throw std::invalid_argument("Unkown random initializer");
                }
            }

            template <typename data_t>
            void InitializerImpl<data_t>::initialize(data_t* data, index_t size,
                                                     Initializer initializer)
            {
                FanPairType fan{0, 0};
                initialize(data, size, initializer, fan);
            }

            template <typename data_t>
            std::mt19937_64 InitializerImpl<data_t>::getEngine()
            {
                if (useSeed_)
                    return std::mt19937_64(seed_);
                else
                    return std::mt19937_64(randomDevice_());
            }

            template <typename data_t>
            void InitializerImpl<data_t>::uniform(data_t* data, index_t size, data_t lowerBound,
                                                  data_t upperBound)
            {
                UniformDistributionType dist(lowerBound, upperBound);
                std::mt19937_64 engine = getEngine();

                for (index_t i = 0; i < size; ++i)
                    data[i] = dist(engine);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::uniform(data_t* data, index_t size)
            {
                InitializerImpl<data_t>::uniform(data, size, 0, std::numeric_limits<data_t>::max());
            }

            template <typename data_t>
            void InitializerImpl<data_t>::constant(data_t* data, index_t size, data_t constant)
            {
                for (index_t i = 0; i < size; ++i)
                    data[i] = constant;
            }

            template <typename data_t>
            void InitializerImpl<data_t>::ones(data_t* data, index_t size)
            {
                constant(data, size, static_cast<data_t>(1));
            }

            template <typename data_t>
            void InitializerImpl<data_t>::zeros(data_t* data, index_t size)
            {
                constant(data, size, static_cast<data_t>(0));
            }

            template <typename data_t>
            void InitializerImpl<data_t>::glorotUniform(
                data_t* data, index_t size, const InitializerImpl<data_t>::FanPairType& fan)
            {
                auto bound = static_cast<data_t>(std::sqrt(
                    6 / (static_cast<data_t>(fan.first) + static_cast<data_t>(fan.second))));
                uniform(data, size, -1 * bound, bound);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::glorotNormal(
                data_t* data, index_t size, const InitializerImpl<data_t>::FanPairType& fan)
            {
                auto stddev = static_cast<data_t>(std::sqrt(
                    2 / (static_cast<data_t>(fan.first) + static_cast<data_t>(fan.second))));
                truncatedNormal(data, size, 0, stddev);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::normal(data_t* data, index_t size, data_t mean,
                                                 data_t stddev)
            {
                static_assert(!std::is_same<std::false_type, NormalDistributionType>::value,
                              "Cannot use normal distribution with the given data-type");

                NormalDistributionType dist(mean, stddev);
                std::mt19937_64 engine = getEngine();

                for (index_t i = 0; i < size; ++i)
                    data[i] = dist(engine);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::truncatedNormal(data_t* data, index_t size, data_t mean,
                                                          data_t stddev)
            {
                static_assert(!std::is_same<std::false_type, NormalDistributionType>::value,
                              "Cannot use normal distribution with the given data-type");

                NormalDistributionType dist(mean, stddev);
                std::mt19937_64 engine = getEngine();

                for (index_t i = 0; i < size; ++i) {
                    auto value = dist(engine);
                    while (std::abs(mean - value) > 2 * stddev) {
                        value = dist(engine);
                    }
                    data[i] = value;
                }
            }

            template <typename data_t>
            void InitializerImpl<data_t>::heNormal(
                data_t* data, index_t size, const InitializerImpl<data_t>::FanPairType& fanInOut)
            {
                auto stddev =
                    std::sqrt(static_cast<data_t>(2) / static_cast<data_t>(fanInOut.first));
                truncatedNormal(data, size, 0, stddev);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::heUniform(data_t* data, index_t size,
                                                    const InitializerImpl<data_t>::FanPairType& fan)
            {
                auto bound = static_cast<data_t>(std::sqrt(6 / (static_cast<data_t>(fan.first))));
                uniform(data, size, -1 * bound, bound);
            }

            template <typename data_t>
            void InitializerImpl<data_t>::ramlak(data_t* data, index_t size)
            {
                const index_t hw = static_cast<data_t>(size - data_t(1)) / data_t(2);

                for (index_t i = -hw; i <= hw; ++i) {
                    if (i % 2)
                        data[i + hw] = data_t(-1)
                                       / (static_cast<data_t>(i) * static_cast<data_t>(i)
                                          * pi<data_t> * pi<data_t>);
                    else
                        data[i + hw] = data_t(0);
                }
                data[hw] = data_t(0.25);
            }

            template class InitializerImpl<float>;
        } // namespace detail
    }     // namespace ml
} // namespace elsa