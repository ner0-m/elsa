/**
 * \file test_RayGenerationBench.cpp
 *
 * \brief Benchmarks for projectors
 *
 * \author David Frank
 */

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>

#include "Logger.h"
#include "elsaDefines.h"
#include "PhantomGenerator.h"
#include "CircleTrajectoryGenerator.h"

#include "SiddonsMethod.h"
#include "JosephsMethod.h"

using namespace elsa;

using DataContainer_t = DataContainer<real_t>;

// Some minor fixtures
template <typename Projector>
void runProjector2D(index_t coeffsPerDim)
{
    // generate 2d phantom
    IndexVector_t size(2);
    size << coeffsPerDim, coeffsPerDim;
    auto phantom = DataContainer_t(DataDescriptor(size));
    phantom = 0;

    // generate circular trajectory
    index_t noAngles{180}, arc{360};
    auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(
        noAngles, phantom.getDataDescriptor(), arc, 20, 20);

    // setup operator for 2d X-ray transform
    Projector projector(phantom.getDataDescriptor(), *sinoDescriptor, geometry);

    DataContainer_t sinogram(*sinoDescriptor);
    BENCHMARK("Forward projection")
    {
        sinogram = projector.apply(phantom);
        return sinogram;
    };

    BENCHMARK("Backward projection") { return projector.applyAdjoint(sinogram); };
}

template <typename Projector>
void runProjector3D(index_t coeffsPerDim)
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    // generate 2d phantom
    IndexVector_t size(3);
    size << coeffsPerDim, coeffsPerDim, coeffsPerDim;
    auto phantom = DataContainer_t(DataDescriptor(size));
    phantom = 0;

    // generate circular trajectory
    index_t noAngles{180}, arc{360};
    auto [geometry, sinoDescriptor] = CircleTrajectoryGenerator::createTrajectory(
        noAngles, phantom.getDataDescriptor(), arc, 20, 20);

    // setup operator for 2d X-ray transform
    Projector projector(phantom.getDataDescriptor(), *sinoDescriptor, geometry);

    DataContainer_t sinogram(*sinoDescriptor);
    BENCHMARK("Forward projection")
    {
        sinogram = projector.apply(phantom);
        return sinogram;
    };

    BENCHMARK("Backward projection") { return projector.applyAdjoint(sinogram); };
}

TEST_CASE("Testing Siddon's projector in 2D")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    using Siddon = SiddonsMethod<real_t>;

    GIVEN("A 8x8 Problem:") { runProjector2D<Siddon>(8); }

    GIVEN("A 16x16 Problem:") { runProjector2D<Siddon>(16); }

    GIVEN("A 32x32 Problem:") { runProjector2D<Siddon>(32); }

    GIVEN("A 64x64 Problem:") { runProjector2D<Siddon>(64); }
}

TEST_CASE("Testing Siddon's projector in 3D")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    using Siddon = SiddonsMethod<real_t>;

    GIVEN("A 8x8x8 Problem:") { runProjector3D<Siddon>(8); }

    GIVEN("A 16x16x16 Problem:") { runProjector3D<Siddon>(16); }

    GIVEN("A 32x32x32 Problem:") { runProjector3D<Siddon>(32); }
}

TEST_CASE("Testing Joseph's projector in 2D")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    using Joseph = JosephsMethod<real_t>;

    GIVEN("A 8x8 Problem:") { runProjector2D<Joseph>(8); }

    GIVEN("A 16x16 Problem:") { runProjector2D<Joseph>(16); }

    GIVEN("A 32x32 Problem:") { runProjector2D<Joseph>(32); }

    GIVEN("A 64x64 Problem:") { runProjector2D<Joseph>(64); }
}

TEST_CASE("Testing Joseph's projector in 3D")
{
    // Turn logger off
    Logger::setLevel(Logger::LogLevel::OFF);

    using Joseph = JosephsMethod<real_t>;

    GIVEN("A 8x8x8 Problem:") { runProjector3D<Joseph>(8); }

    GIVEN("A 16x16x16 Problem:") { runProjector3D<Joseph>(16); }

    GIVEN("A 32x32x32 Problem:") { runProjector3D<Joseph>(32); }
}