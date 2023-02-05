#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include "elsa.h"
#include "spdlog/fmt/fmt.h"

template <template <class> class Projector, class data_t, bool forward>
void bench(const std::string& name, elsa::IndexVector_t size, ankerl::nanobench::Bench& b)
{
    auto phantom = elsa::phantoms::modifiedSheppLogan<data_t>(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    // generate circular trajectory
    elsa::index_t numAngles{512}, arc{360};
    const auto distance = static_cast<data_t>(size(0));
    auto sinoDescriptor = elsa::CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, distance * 100.0f, distance);

    Projector<data_t> projector(dynamic_cast<const elsa::VolumeDescriptor&>(volumeDescriptor),
                                *sinoDescriptor);

    if constexpr (forward) {
        b.run(name, [&]() {
            auto sino = projector.apply(phantom);
            ankerl::nanobench::doNotOptimizeAway(sino);
        });
    } else {
        auto sino = projector.apply(phantom);
        b.run(name, [&]() {
            auto back = projector.apply(sino);
            ankerl::nanobench::doNotOptimizeAway(back);
        });
    }
}

ankerl::nanobench::Bench getNewBench()
{

    using namespace std::chrono_literals;

    ankerl::nanobench::Bench b;
    b.title("Projections");
    b.performanceCounters(true);
    b.minEpochIterations(20);
    b.timeUnit(1ms, "ms");
    b.relative(true);
    return b;
}

int main()
{
    elsa::Logger::setLevel(elsa::Logger::LogLevel::OFF);
    ankerl::nanobench::Bench b;

#define ELSA_BENCHMARK_TYPE_SIZE(projector, type, size, forward)                              \
    bench<projector, type, forward>(                                                          \
        fmt::format("{} {} 2D {}x{} forward: {}", #projector, #type, #size, #size, #forward), \
        elsa::IndexVector_t({{size, size}}), b);

#define ELSA_BENCHMARK_SIZE(projector, size, forward) \
    ELSA_BENCHMARK_TYPE_SIZE(projector, float, size, forward)

    // #define ELSA_BENCHMARK(projector) ELSA_BENCHMARK_SIZE(projector, 128)

#define ELSA_BENCHMARK_COMPARE(projector1, projector2, size) \
    b = getNewBench();                                       \
    ELSA_BENCHMARK_SIZE(projector1, size, true)              \
    ELSA_BENCHMARK_SIZE(projector2, size, true)              \
    b = getNewBench();                                       \
    ELSA_BENCHMARK_SIZE(projector1, size, false)             \
    ELSA_BENCHMARK_SIZE(projector2, size, false)

    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 32)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 32)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 50)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 50)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 64)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 64)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 100)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 100)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 128)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 128)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 200)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 200)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 256)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 256)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 512)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 512)
    ELSA_BENCHMARK_COMPARE(elsa::JosephsMethod, elsa::JosephsMethodBranchless, 1024)
    ELSA_BENCHMARK_COMPARE(elsa::SiddonsMethod, elsa::SiddonsMethodBranchless, 1024)
}
