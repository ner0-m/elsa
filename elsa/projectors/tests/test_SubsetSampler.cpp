/**
 * @file test_SubsetSampler.cpp
 *
 * @brief Tests for the SubsetSampler class
 *
 * @author Michael Loipführer - initial code
 */
#include <SphereTrajectoryGenerator.h>
#include "doctest/doctest.h"

#include "SubsetSampler.h"
#include "Logger.h"
#include "CircleTrajectoryGenerator.h"
#include "SiddonsMethod.h"
#include "PlanarDetectorDescriptor.h"
#include "PhantomGenerator.h"
#include "SiddonsMethod.h"
#include "JosephsMethod.h"

using namespace elsa;
using namespace doctest;

SCENARIO("Testing the standalone subset sampling strategies")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("A circular trajectory with 32 angles")
    {
        IndexVector_t size(2);
        size << 128, 128;
        VolumeDescriptor volumeDescriptor{size};
        index_t numAngles{32}, arc{360};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, volumeDescriptor, arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));

        WHEN("performing round robin sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            const std::vector<std::vector<index_t>> mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::sampleRoundRobin(
                    static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                for (std::size_t i = 0; i < static_cast<std::size_t>(nSubsets); ++i) {
                    REQUIRE_EQ(mapping[i].size(), 8);
                    for (std::size_t j = 0; j < mapping[i].size(); j++) {
                        REQUIRE_EQ(mapping[i][j], j * nSubsets + i);
                    }
                }
            }
        }
        WHEN("performing equi rotation sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            const std::vector<std::vector<index_t>> mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::sampleEquiRotation(
                    static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                REQUIRE_EQ(mapping[0], std::vector<index_t>{0, 3, 7, 11, 15, 19, 23, 27});
                REQUIRE_EQ(mapping[1], std::vector<index_t>{31, 4, 8, 12, 16, 20, 24, 28});
                REQUIRE_EQ(mapping[2], std::vector<index_t>{1, 5, 9, 13, 17, 21, 25, 29});
                REQUIRE_EQ(mapping[3], std::vector<index_t>{2, 6, 10, 14, 18, 22, 26, 30});
            }
        }
    }

    GIVEN("A spherical trajectory with 32 angles and 4 circles")
    {
        IndexVector_t size(3);
        size << 128, 128, 128;
        VolumeDescriptor volumeDescriptor{size};
        index_t numPoses{32}, numCircles{4};
        auto sinoDescriptor = SphereTrajectoryGenerator::createTrajectory(
            numPoses, volumeDescriptor, numCircles,
            geometry::SourceToCenterOfRotation(static_cast<real_t>(size(0)) * 100.0f),
            geometry::CenterOfRotationToDetector(static_cast<real_t>(size(0))));

        WHEN("performing round robin sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            const auto mapping = SubsetSampler<PlanarDetectorDescriptor, real_t>::sampleRoundRobin(
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                for (std::size_t i = 0; i < static_cast<std::size_t>(nSubsets); ++i) {
                    REQUIRE_EQ(mapping[i].size(), 8);
                    for (std::size_t j = 0; j < mapping[i].size(); j++) {
                        REQUIRE_EQ(mapping[i][j], j * nSubsets + i);
                    }
                }
            }
        }
        WHEN("performing equi rotation sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            const auto mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::sampleEquiRotation(
                    static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                REQUIRE_EQ(mapping[0], std::vector<index_t>{0, 22, 1, 24, 26, 2, 29, 11});
                REQUIRE_EQ(mapping[1], std::vector<index_t>{4, 13, 7, 23, 27, 9, 30, 10});
                REQUIRE_EQ(mapping[2], std::vector<index_t>{12, 5, 15, 14, 17, 18, 31, 3});
                REQUIRE_EQ(mapping[3], std::vector<index_t>{21, 6, 25, 16, 8, 28, 20, 19});
            }
        }
    }
}

SCENARIO("Testing SubsetSampler with PlanarDetectorDescriptor and circular trajectory")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    index_t numAngles{180}, arc{360};
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100.0f,
        static_cast<real_t>(size(0)));

    SiddonsMethod projector(static_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    auto sinogram = projector.apply(phantom);
    const auto coeffsPerDimSinogram =
        sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension();

    GIVEN("A small phantom problem")
    {

        WHEN("Setting up a subset sampler with 4 subsets")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("the clone works as expected")
            {

                auto subsetSamplerClone = subsetSampler.clone();

                REQUIRE_NE(subsetSamplerClone.get(), &subsetSampler);
                REQUIRE_EQ(*subsetSamplerClone, subsetSampler);
            }
            AND_THEN("The full data has the correct dimensions")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfDimensions(),
                           sinogram.getDataDescriptor().getNumberOfDimensions());
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfCoefficientsPerDimension(),
                           coeffsPerDimSinogram);
            }

            AND_THEN("It returns the correct data blocks and projectors for each subsets")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                for (index_t i = 0; i < nSubsets; i++) {
                    const auto subset = data.getBlock(i);
                    const auto coeffsPerDimension =
                        subset.getDataDescriptor().getNumberOfCoefficientsPerDimension();

                    REQUIRE_EQ(coeffsPerDimension[0], coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                REQUIRE_EQ(subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>().size(),
                           nSubsets);
            }
        }
        WHEN("Setting up a subset sampler with 4 subsets and EQUI_ROTATION sampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets,
                SubsetSampler<PlanarDetectorDescriptor, real_t>::SamplingStrategy::EQUI_ROTATION);
            THEN("the clone works as expected")
            {

                auto subsetSamplerClone = subsetSampler.clone();

                REQUIRE_NE(subsetSamplerClone.get(), &subsetSampler);
                REQUIRE_EQ(*subsetSamplerClone, subsetSampler);
            }
            AND_THEN("The full data has the correct dimensions")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfDimensions(),
                           sinogram.getDataDescriptor().getNumberOfDimensions());
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfCoefficientsPerDimension(),
                           coeffsPerDimSinogram);
            }

            AND_THEN("It returns the correct data blocks and projectors for each subsets")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                for (index_t i = 0; i < nSubsets; i++) {
                    const auto subset = data.getBlock(i);
                    const auto coeffsPerDimension =
                        subset.getDataDescriptor().getNumberOfCoefficientsPerDimension();

                    REQUIRE_EQ(coeffsPerDimension[0], coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                REQUIRE_EQ(subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>().size(),
                           nSubsets);
            }
        }
    }
}

SCENARIO("Testing SubsetSampler with PlanarDetectorDescriptor and spherical trajectory")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    IndexVector_t size(3);
    size << 32, 32, 32;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    index_t numPoses{180}, numCircles{5};
    auto sinoDescriptor = SphereTrajectoryGenerator::createTrajectory(
        numPoses, phantom.getDataDescriptor(), numCircles,
        geometry::SourceToCenterOfRotation(static_cast<real_t>(size(0)) * 100.0f),
        geometry::CenterOfRotationToDetector(static_cast<real_t>(size(0))));

    SiddonsMethod projector(static_cast<const VolumeDescriptor&>(volumeDescriptor),
                            *sinoDescriptor);

    auto sinogram = projector.apply(phantom);
    const auto coeffsPerDimSinogram =
        sinogram.getDataDescriptor().getNumberOfCoefficientsPerDimension();

    GIVEN("A small phantom problem")
    {

        WHEN("Setting up a subset sampler with 4 subsets")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("the clone works as expected")
            {

                auto subsetSamplerClone = subsetSampler.clone();

                REQUIRE_NE(subsetSamplerClone.get(), &subsetSampler);
                REQUIRE_EQ(*subsetSamplerClone, subsetSampler);
            }
            AND_THEN("The full data has the correct dimensions")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfDimensions(),
                           sinogram.getDataDescriptor().getNumberOfDimensions());
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfCoefficientsPerDimension(),
                           coeffsPerDimSinogram);
            }

            AND_THEN("It returns the correct data blocks and projectors for each subsets")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                for (index_t i = 0; i < nSubsets; i++) {
                    const auto subset = data.getBlock(i);
                    const auto coeffsPerDimension =
                        subset.getDataDescriptor().getNumberOfCoefficientsPerDimension();

                    REQUIRE(coeffsPerDimension[0] == coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                REQUIRE_EQ(subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>().size(),
                           nSubsets);
            }
        }
        WHEN("Setting up a subset sampler with 4 subsets and EQUI_ROTATION sampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets,
                SubsetSampler<PlanarDetectorDescriptor, real_t>::SamplingStrategy::EQUI_ROTATION);
            THEN("the clone works as expected")
            {

                auto subsetSamplerClone = subsetSampler.clone();

                REQUIRE_NE(subsetSamplerClone.get(), &subsetSampler);
                REQUIRE_EQ(*subsetSamplerClone, subsetSampler);
            }
            AND_THEN("The full data has the correct dimensions")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfDimensions(),
                           sinogram.getDataDescriptor().getNumberOfDimensions());
                REQUIRE_EQ(data.getDataDescriptor().getNumberOfCoefficientsPerDimension(),
                           coeffsPerDimSinogram);
            }

            AND_THEN("It returns the correct data blocks and projectors for each subsets")
            {
                const auto data = subsetSampler.getPartitionedData(sinogram);
                for (index_t i = 0; i < nSubsets; i++) {
                    const auto subset = data.getBlock(i);
                    const auto coeffsPerDimension =
                        subset.getDataDescriptor().getNumberOfCoefficientsPerDimension();

                    REQUIRE_EQ(coeffsPerDimension[0], coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                REQUIRE_EQ(subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>().size(),
                           nSubsets);
            }
        }
    }
}
