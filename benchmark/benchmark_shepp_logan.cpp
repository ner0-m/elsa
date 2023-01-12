#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include "elsa.h"
#include "spdlog/fmt/fmt.h"
#include <iostream>
#include <fstream>

static int ITERATIONS = 20;

using namespace std;
void bench(std::vector<elsa::index_t>& sizes, ofstream& file)
{

    using namespace std::chrono_literals;

    ankerl::nanobench::Bench b;
    b.title("Shepp Logan");
    b.performanceCounters(true);
    b.minEpochIterations(ITERATIONS);
    b.timeUnit(1ms, "ms");

    for (auto size : sizes) {
        auto name = fmt::format("{}³ Shepp Logan", size);
        elsa::IndexVector_t sizes{{size, size, size}};
        b.run(name + " New", [&]() {
            ankerl::nanobench::doNotOptimizeAway(elsa::phantoms::modifiedSheppLogan<double>(sizes));
        });
        b.run(name + " Old", [&]() {
            ankerl::nanobench::doNotOptimizeAway(
                elsa::phantoms::old::modifiedSheppLogan<double>(sizes));
        });
    }

    ankerl::nanobench::render(ankerl::nanobench::templates::json(), b, file);
}

int main()
{
    elsa::Logger::setLevel(elsa::Logger::LogLevel::OFF);
    int inc = 30;
    std::vector<elsa::index_t> sizes(1200 / inc);
    std::generate(sizes.begin(), sizes.end(), [n = 1, &inc]() mutable { return inc * n++; });
    ofstream f;
    f.open("shepp_logan.json");
    f << "[";
    bench(sizes, f);
    f << "]";
}
