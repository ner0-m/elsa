/// Elsa example program: test execution speed of GPU projectors

#include "elsa.h"

#include <chrono>
#include <iostream>
#include <vector>

using namespace elsa;

void testExecutionSpeed(LinearOperator<real_t>& projector, DataContainer<real_t>& volume,
                        int numIters)
{
    real_t timer = 0;

    // set up containers for results of forward and backward projection
    DataContainer projections(projector.getRangeDescriptor());
    DataContainer backproj(projector.getDomainDescriptor());

    // run test for forward projection
    for (int i = 0; i < numIters; ++i) {
        auto start = std::chrono::system_clock::now();
        projector.apply(volume, projections);
        auto stop = std::chrono::system_clock::now();
        timer += static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    }

    // log average execution time
    timer /= static_cast<float>(numIters);
    Logger::get("Timing")->info("average apply time: {}\n", timer);

    timer = 0;

    // run test for backward projection
    for (int i = 0; i < numIters; ++i) {
        auto start = std::chrono::system_clock::now();
        projector.applyAdjoint(projections, backproj);
        auto stop = std::chrono::system_clock::now();
        timer += static_cast<float>(
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    }

    // log average execution time
    timer /= static_cast<float>(numIters);
    Logger::get("Timing")->info("average apply adjoint time: {}\n", timer);
}

int main()
{
    Logger::setLevel(Logger::LogLevel::INFO);

    int numAngles = 512;

    // volume sizes to be tested
    std::vector<int> sizes{128, 256, 384, 512};

    // each operation will be applied numIters times, and the average displayed
    int numIters = 3;

    for (int size : sizes) {
        Logger::get("Setup")->info("Running test for a volume with {}^3 voxels\n", size);

        // create 3d phantom
        IndexVector_t volumeSize(3);
        volumeSize << size, size, size;
        auto phantom = phantoms::modifiedSheppLogan(volumeSize);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        // generate circular trajectory with numAngles angles over 360 degrees
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, volumeDescriptor, 360, 30.0f * static_cast<real_t>(size),
            2.0f * static_cast<real_t>(size));

        // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns
        // a VolumeDescriptor
        auto& volDescriptor = downcast<VolumeDescriptor>(volumeDescriptor);

        // setup and run test for fast Joseph's
        Logger::get("Setup")->info("Fast unmatched Joseph's:\n");
        auto josephsFast = JosephsMethodCUDA(volDescriptor, *sinoDescriptor);
        testExecutionSpeed(josephsFast, phantom, numIters);

        // setup and run test for slow Joseph's
        Logger::get("Setup")->info("Slow matched Joseph's:\n");
        auto josephsSlow = JosephsMethodCUDA(volDescriptor, *sinoDescriptor, false);
        testExecutionSpeed(josephsSlow, phantom, numIters);

        // setup and run test for Siddon's
        Logger::get("Setup")->info("Siddon's:\n");
        auto siddons = SiddonsMethodCUDA(volDescriptor, *sinoDescriptor);
        testExecutionSpeed(siddons, phantom, numIters);
    }
}
