#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include "elsa.h"
#include "spdlog/fmt/fmt.h"
#include <vector>
#include <iostream>
#include <fstream>

static int ITERATIONS = 20;
using namespace std;

template <typename data_t>
void benchHead(std::vector<elsa::index_t>& sizes, ofstream& file)
{
    using namespace std::chrono_literals;
    ankerl::nanobench::Bench b;
    b.title("Forbild head");
    b.performanceCounters(true);
    b.minEpochIterations(ITERATIONS);
    b.timeUnit(1ms, "ms");
    for (auto s : sizes) {
        elsa::IndexVector_t size(3);
        size << s, s, s;
        b.run(fmt::format("{}³ FORBILD Thorax [{}]", s, typeid(data_t).name()), [&]() {
            ankerl::nanobench::doNotOptimizeAway(elsa::phantoms::forbild_head<data_t>(size));
        });
    }

    ankerl::nanobench::render(ankerl::nanobench::templates::json(), b, file);
}

template <typename data_t>
void benchAbdomen(std::vector<elsa::index_t>& sizes, ofstream& file)
{
    using namespace std::chrono_literals;
    ankerl::nanobench::Bench b;
    b.title("Forbild abdomen");
    b.performanceCounters(true);
    b.minEpochIterations(ITERATIONS);
    b.timeUnit(1ms, "ms");
    for (auto s : sizes) {
        elsa::IndexVector_t size(3);
        size << s, s, s;
        b.run(fmt::format("{}³ FORBILD Thorax [{}]", s, typeid(data_t).name()), [&]() {
            ankerl::nanobench::doNotOptimizeAway(elsa::phantoms::forbild_abdomen<data_t>(size));
        });
    }

    ankerl::nanobench::render(ankerl::nanobench::templates::json(), b, file);
}

template <typename data_t>
void benchThorax(std::vector<elsa::index_t>& sizes, ofstream& file)
{
    using namespace std::chrono_literals;
    ankerl::nanobench::Bench b;
    b.title("Forbild thorax");
    b.performanceCounters(true);
    b.minEpochIterations(ITERATIONS);
    b.timeUnit(1ms, "ms");
    for (auto s : sizes) {
        elsa::IndexVector_t size(3);
        size << s, s, s;
        b.run(fmt::format("{}³ FORBILD Thorax [{}]", s, typeid(data_t).name()), [&]() {
            ankerl::nanobench::doNotOptimizeAway(elsa::phantoms::forbild_thorax<data_t>(size));
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
    f.open("forbild.json");
    f << "[";
    benchHead<double>(sizes, f);
    f << ",";
    benchAbdomen<double>(sizes, f);
    f << ",";
    benchThorax<double>(sizes, f);
    f << "]";
    f.close();
}
