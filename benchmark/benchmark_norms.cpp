#include <cmath>
#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>

#include "Eigen/Core"
#include "spdlog/fmt/fmt.h"

#include "VolumeDescriptor.h"
#include "DataContainer.h"
#include "elsaDefines.h"

void bench(elsa::index_t dim, elsa::index_t s)
{
    ankerl::nanobench::Bench b;
    b.title(fmt::format("{}D, {}", dim, s));
    b.performanceCounters(true);
    b.minEpochIterations(200);
    b.relative(true);

    {
        auto setup = [&]() {
            auto size = elsa::IndexVector_t(dim);
            size.setConstant(s);

            auto desc = elsa::VolumeDescriptor(size);

            auto data = elsa::Vector_t<float>::Random(size.prod());
            return elsa::DataContainer<float>(desc, data);
        };
        const auto dc = setup();

        b.run("float  l-2", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.l2Norm()); });
        b.run("float  l-1", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.l1Norm()); });
        b.run("float  l-infinity", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.lInfNorm()); });
        b.run("float  sum", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.sum()); });
        b.run("float  minElement",
              [&]() { ankerl::nanobench::doNotOptimizeAway(dc.minElement()); });
        b.run("float  maxElement",
              [&]() { ankerl::nanobench::doNotOptimizeAway(dc.maxElement()); });
    }

    {
        auto setup = [&]() {
            auto size = elsa::IndexVector_t(dim);
            size.setConstant(s);

            auto desc = elsa::VolumeDescriptor(size);

            auto data = elsa::Vector_t<double>::Random(size.prod());
            return elsa::DataContainer<double>(desc, data);
        };
        const auto dc = setup();

        b.run("double l-2", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.l2Norm()); });
        b.run("double l-1", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.l1Norm()); });
        b.run("double l-infinity", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.lInfNorm()); });
        b.run("double sum", [&]() { ankerl::nanobench::doNotOptimizeAway(dc.sum()); });
        b.run("double minElement",
              [&]() { ankerl::nanobench::doNotOptimizeAway(dc.minElement()); });
        b.run("double maxElement",
              [&]() { ankerl::nanobench::doNotOptimizeAway(dc.maxElement()); });
    }
}

int main()
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);

#define ELSA_BENCHMARK_NORM(dim, size) bench(dim, size);

    ELSA_BENCHMARK_NORM(2, 128);
    ELSA_BENCHMARK_NORM(2, 256);
    ELSA_BENCHMARK_NORM(2, 512);
    ELSA_BENCHMARK_NORM(2, 1024);
    ELSA_BENCHMARK_NORM(2, 2 * 1024);
    ELSA_BENCHMARK_NORM(3, 32);
    ELSA_BENCHMARK_NORM(3, 64);
    ELSA_BENCHMARK_NORM(3, 128);
    ELSA_BENCHMARK_NORM(3, 256);
    ELSA_BENCHMARK_NORM(3, 512);
}
