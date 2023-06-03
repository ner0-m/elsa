#include <cmath>
#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>

#include "Eigen/Core"
#include "spdlog/fmt/fmt.h"

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "elsaDefines.h"

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
    return "complex float";
}

template <>
std::string nameOf<double>()
{
    return "complex double";
}

template <typename T>
void bench(elsa::index_t dim, elsa::index_t s)
{
    ankerl::nanobench::Bench b;
    b.title(fmt::format("{}D, {}", dim, s));
    b.performanceCounters(true);
    b.minEpochIterations(5);
    b.relative(true);

    {
        auto setup = [&]() {
            std::random_device r;

            std::default_random_engine e(r());
            std::uniform_real_distribution<T> uniform_dist;

            auto size = elsa::IndexVector_t(dim);
            size.setConstant(s);

            auto desc = elsa::VolumeDescriptor(size);

            auto dc = elsa::DataContainer<elsa::complex<T>>(desc);
            thrust::generate(thrust::host, dc.begin(), dc.end(), [&]() {
                elsa::complex<T> c;
                c.real(uniform_dist(e));
                c.imag(uniform_dist(e));
                return c;
            });
            return dc;
        };
        auto dc = setup();

        b.run(nameOf<T>() + "  fft (FORWARD norm)", [&]() { dc.fft(elsa::FFTNorm::FORWARD); });
        b.run(nameOf<T>() + "  ifft (BACKWARD norm)", [&]() { dc.ifft(elsa::FFTNorm::BACKWARD); });

        b.run(nameOf<T>() + "  fft (ORTHO norm)", [&]() { dc.fft(elsa::FFTNorm::ORTHO); });
        b.run(nameOf<T>() + "  ifft (ORTHO norm)", [&]() { dc.ifft(elsa::FFTNorm::ORTHO); });

        ankerl::nanobench::doNotOptimizeAway(dc);
    }
}

int main()
{
    // Set seed for randomized data containers!
    srand((unsigned int) 666);

#define ELSA_BENCHMARK_NORM(dim, size) \
    bench<float>(dim, size);           \
    bench<double>(dim, size);

    ELSA_BENCHMARK_NORM(1, 32);
    ELSA_BENCHMARK_NORM(1, 64);
    ELSA_BENCHMARK_NORM(1, 128);
    ELSA_BENCHMARK_NORM(1, 256);
    ELSA_BENCHMARK_NORM(1, 512);
    ELSA_BENCHMARK_NORM(1, 1024);
    ELSA_BENCHMARK_NORM(1, 2048);
    ELSA_BENCHMARK_NORM(1, 4096);
    ELSA_BENCHMARK_NORM(1, 1 << 20 /* sizeof(TYPE) * 2 * 1MiB */);
    ELSA_BENCHMARK_NORM(2, 256);
    ELSA_BENCHMARK_NORM(2, 512);
    ELSA_BENCHMARK_NORM(2, 1024);
    ELSA_BENCHMARK_NORM(2, 2048);
    ELSA_BENCHMARK_NORM(3, 64);
    ELSA_BENCHMARK_NORM(3, 128);
    ELSA_BENCHMARK_NORM(3, 256);
    ELSA_BENCHMARK_NORM(3, 512);

    /* try some sizes that are not powers of 2 */

    /* this is a prime, which should not work well for cuFFT */
    ELSA_BENCHMARK_NORM(1, 27644437);
    /* sizeof(TYPE) * 2 * 100MB */
    ELSA_BENCHMARK_NORM(1, 100000000);

    /* sizeof(TYPE) * 2 * 100MB*/
    ELSA_BENCHMARK_NORM(2, 10000);

    /* sizeof(TYPE) * 2 * 125MB */
    ELSA_BENCHMARK_NORM(3, 500);
}
