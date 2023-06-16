#include <cmath>
#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>

#include "spdlog/fmt/fmt.h"
#include "elsaDefines.h"
#include "DataContainer.h"

#include "reductions/DotProduct.h"
#include "transforms/InplaceMul.h"

#include <thrust/generate.h>
#include <thrust/complex.h>
#include <thrust/execution_policy.h>

template <typename T>
std::string nameOf()
{
    return "Unknown";
}

template <>
std::string nameOf<float>()
{
    return "float";
}

template <>
std::string nameOf<double>()
{
    return "double";
}

template <>
std::string nameOf<thrust::complex<float>>()
{
    return "complex double";
}

template <>
std::string nameOf<thrust::complex<double>>()
{
    return "complex double";
}

template <typename T>
void bench(size_t size)
{
    using BareType = std::conditional_t<std::is_same<T, thrust::complex<float>>::value || std::is_same<T, float>::value, float, double>;

    ankerl::nanobench::Bench b;
    b.title(fmt::format("{}", size));
    b.performanceCounters(true);
    b.minEpochIterations(5);
    b.relative(true);

    {
        std::random_device r;
        std::default_random_engine e(r());
        std::uniform_real_distribution<BareType> uniform_dist;

        elsa::ContiguousStorage<T> v0(size);
        elsa::ContiguousStorage<T> v1(size);

        auto single = [&]() -> T {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
                return uniform_dist(e);
            else
                return T{uniform_dist(e), uniform_dist(e)};
        };

        thrust::generate(thrust::host, v0.begin(), v0.end(), [&]() -> T { return single(); });
        v1 = v0;
        
        T r0 = T(), r1 = T();
        T scalar = single();
        b.run(nameOf<T>() + "  thrust::dot", [&]() { r0 = elsa::dot(v0.begin(), v0.end(), v1.begin()); });
        b.run(nameOf<T>() + "  cublas::dot", [&]() { r1 = elsa::dot_v2(v0.begin(), v0.end(), v1.begin()); });
        b.run(nameOf<T>() + "  thrust::inplace-mul", [&]() { elsa::inplaceMulScalar(v0.begin(), v0.end(), scalar); });
        b.run(nameOf<T>() + "  cublas::inplace-mul", [&]() { elsa::inplaceMulScalar_v2(v1.begin(), v1.end(), scalar); });
        std::cout << "results: " << r0 << " vs " << r1 << " -> " << (r1 - r0) << std::endl;

        ankerl::nanobench::doNotOptimizeAway(v0);
        ankerl::nanobench::doNotOptimizeAway(v1);
        ankerl::nanobench::doNotOptimizeAway(r0);
        ankerl::nanobench::doNotOptimizeAway(r1);
    }
}

int main()
{
    // Set seed for randomized data containers!
    srand((unsigned int) 666);

#define ELSA_BENCHMARK_NORM(size)        \
    bench<float>(size);                  \
    bench<double>(size);                 \
    bench<thrust::complex<float>>(size); \
    bench<thrust::complex<double>>(size);

    ELSA_BENCHMARK_NORM(32);
    ELSA_BENCHMARK_NORM(64);
    ELSA_BENCHMARK_NORM(128);
    ELSA_BENCHMARK_NORM(256);
    ELSA_BENCHMARK_NORM(512);
    ELSA_BENCHMARK_NORM(1024);
    ELSA_BENCHMARK_NORM(2048);
    ELSA_BENCHMARK_NORM(4096);
    ELSA_BENCHMARK_NORM(1 << 20 /* sizeof(TYPE) * 2 * 1MiB */);
    
    /* this is a prime, which should not work well for cuFFT */
    ELSA_BENCHMARK_NORM(27644437);
    /* sizeof(TYPE) * 2 * 100MB */
    ELSA_BENCHMARK_NORM(100000000);

    /* sizeof(TYPE) * 2 * 100MB*/
    ELSA_BENCHMARK_NORM(10000);
}
