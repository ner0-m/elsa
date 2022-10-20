#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include "elsa.h"
#include "spdlog/fmt/fmt.h"

template <template <class> class Projector, class data_t>
void bench(const std::string& name, elsa::IndexVector_t size)
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

    using namespace std::chrono_literals;

    ankerl::nanobench::Bench b;
    b.title("Forward projections");
    b.performanceCounters(true);
    b.minEpochIterations(20);
    b.timeUnit(1ms, "ms");
    b.run(name, [&]() {
        auto sino = projector.apply(phantom);
        ankerl::nanobench::doNotOptimizeAway(sino);
    });
}

int main()
{
    elsa::Logger::setLevel(elsa::Logger::LogLevel::OFF);

#define ELSA_BENCHMARK_TYPE_SIZE(projector, type, size)                                    \
    bench<projector, type>(fmt::format("{} {} 2D {}x{}", #projector, #type, #size, #size), \
                           elsa::IndexVector_t({{size, size}}));

#define ELSA_BENCHMARK_SIZE(projector, size)         \
    ELSA_BENCHMARK_TYPE_SIZE(projector, float, size) \
    ELSA_BENCHMARK_TYPE_SIZE(projector, double, size)

#define ELSA_BENCHMARK(projector)       \
    ELSA_BENCHMARK_SIZE(projector, 32)  \
    ELSA_BENCHMARK_SIZE(projector, 50)  \
    ELSA_BENCHMARK_SIZE(projector, 64)  \
    ELSA_BENCHMARK_SIZE(projector, 100) \
    ELSA_BENCHMARK_SIZE(projector, 128) \
    ELSA_BENCHMARK_SIZE(projector, 200) \
    ELSA_BENCHMARK_SIZE(projector, 256)

    ELSA_BENCHMARK(elsa::SiddonsMethod)
    ELSA_BENCHMARK(elsa::JosephsMethod)

#ifdef ELSA_HAS_CUDA_PROJECTORS
    ELSA_BENCHMARK(elsa::SiddonsMethodCUDA)
    ELSA_BENCHMARK(elsa::JosephsMethodCUDA)
#endif
}
