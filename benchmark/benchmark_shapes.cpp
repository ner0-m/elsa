#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include "elsa.h"
#include "spdlog/fmt/fmt.h"
#include <vector>
#include "ForbildPhantom.h"
#include <iostream>
#include <fstream>

using namespace std::chrono_literals;
using namespace elsa::phantoms;
using namespace elsa;

using namespace std;
static int ITERATIONS = 20;

template <typename data_t>
void bench(index_t DIMENSION, ofstream& file)
{
    ankerl::nanobench::Bench b;
    b.title("Forbild head");
    b.performanceCounters(true);
    b.minEpochIterations(ITERATIONS);
    b.timeUnit(1ms, "ms");

    elsa::IndexVector_t size(3);
    size << DIMENSION, DIMENSION, DIMENSION;
    VolumeDescriptor dd(size);
    DataContainer<data_t> dc(dd);
    dc = 0;

    Vec3i center;
    center << DIMENSION / 2, DIMENSION / 2, DIMENSION / 2;

    Vec3X<double> edgeLength;
    edgeLength << DIMENSION - 2, DIMENSION - 2, DIMENSION - 2;

    data_t halfDim = data_t(DIMENSION / 2) - 1.0;

    Vec3X<double> eulers;
    eulers << 0.0, 0.0, 0.0;

    Vec3X<double> halfAxis;
    halfAxis << halfDim, halfDim, halfDim;

    Box box{1.0, center, edgeLength};
    b.run(fmt::format("Box {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(box, dd, dc); });

    Sphere sphere{1.0, center, halfDim};
    b.run(fmt::format("Sphere {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(sphere, dd, dc); });

    Ellipsoid ellip{1.0, center, halfAxis, eulers};
    b.run(fmt::format("Ellipsoid no Rotation {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(ellip, dd, dc); });

    Cylinder cyl{Orientation::X_AXIS, 1.0, center, halfDim, data_t(DIMENSION)};
    b.run(fmt::format("Cylinder on x axis {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(cyl, dd, dc); });

    eulers << 90.0, 40.0, 30.0;
    Ellipsoid ellipRot{1.0, center, halfAxis, eulers};
    b.run(fmt::format("Ellipsoid rotation {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(ellipRot, dd, dc); });

    Vec2X<data_t> halfAxis2(2);
    halfAxis2 << halfDim, halfDim;
    EllipCylinderFree ecyl{1.0, center, halfAxis2, halfDim / 2, eulers};
    b.run(fmt::format("EllipCylinderFree on x axis {}³ [{}]", DIMENSION, typeid(data_t).name()),
          [&]() { rasterize<Blending::ADDITION, data_t>(ecyl, dd, dc); });

    ankerl::nanobench::render(ankerl::nanobench::templates::json(), b, file);
}

int main()
{
    elsa::Logger::setLevel(elsa::Logger::LogLevel::OFF);

    int inc = 100;
    std::vector<elsa::index_t> sizes(1200 / inc);
    std::generate(sizes.begin(), sizes.end(), [n = 1, &inc]() mutable { return inc * n++; });
    ofstream f;
    f.open("shapes.json");
    f << "[";
    for (auto a : sizes) {
        bench<double>(a, f);
        f << ",";
    }
    f << "]";
}
