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
    b.minEpochIterations(300);
    b.relative(true);

    auto setup = [](auto dim, auto s) {
        auto size = elsa::IndexVector_t(dim);
        size.setConstant(s);

        auto desc = elsa::VolumeDescriptor(size);

        auto data = elsa::Vector_t<float>::Random(size.prod());
        return elsa::DataContainer<float>(desc, data);
    };

    const auto x = setup(dim, s);
    const auto y = setup(dim, s);
    auto out = elsa::DataContainer(x.getDataDescriptor());

    b.run("naive: 1 * x + 1 * y", [&]() { ankerl::nanobench::doNotOptimizeAway(x + y); });
    b.run("naive: 2 * x + 1 * y", [&]() { ankerl::nanobench::doNotOptimizeAway(2 * x + y); });
    b.run("naive: 2 * x + 5 * y", [&]() { ankerl::nanobench::doNotOptimizeAway(2 * x + 5 * y); });
    b.run("naive: 1 * x - 1 * y", [&]() { ankerl::nanobench::doNotOptimizeAway(x - y); });
    b.run("optimized: 1 * x + 1 * y",
          [&]() { ankerl::nanobench::doNotOptimizeAway(elsa::lincomb(1, x, 1, y)); });
    b.run("optimized: 2 * x + 1 * y",
          [&]() { ankerl::nanobench::doNotOptimizeAway(elsa::lincomb(2, x, 1, y)); });
    b.run("optimized: 2 * x + 5 * y",
          [&]() { ankerl::nanobench::doNotOptimizeAway(elsa::lincomb(2, x, 5, y)); });
    b.run("optimized: 1 * x - 1 * y",
          [&]() { ankerl::nanobench::doNotOptimizeAway(elsa::lincomb(1, x, -1, y)); });

    b.run("naive(out): 1 * x + 1 * y", [&]() {
        out = x + y;
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("naive(out): 2 * x + 1 * y", [&]() {
        out = 2 * x + y;
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("naive(out): 2 * x + 5 * y", [&]() {
        out = 2 * x + 5 * y;
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("naive(out): 1 * x - 1 * y", [&]() {
        out = x - y;
        ankerl::nanobench::doNotOptimizeAway(out);
    });

    b.run("optimized(out): 1 * x + 1 * y", [&]() {
        elsa::lincomb(1, x, 1, y, out);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("optimized(out): 2 * x + 1 * y", [&]() {
        elsa::lincomb(2, x, 1, y, out);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("optimized(out): 2 * x + 5 * y", [&]() {
        elsa::lincomb(2, x, 5, y, out);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    b.run("optimized(out): 1 * x - 1 * y", [&]() {
        elsa::lincomb(1, x, -1, y, out);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

int main()
{
    // Set seed for Eigen Matrices!
    srand((unsigned int) 666);
    bench(2, 128);
    bench(2, 256);
    bench(2, 512);
    bench(2, 1024);

    bench(3, 64);
    bench(3, 128);
    bench(3, 256);
    bench(3, 512);
}
