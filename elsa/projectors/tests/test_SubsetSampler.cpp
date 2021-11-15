/**
 * @file test_SubsetSampler.cpp
 *
 * @brief Tests for the SubsetSampler class
 *
 * @author Michael Loipführer - initial code
 */
#include <SphereTrajectoryGenerator.h>
#include <numeric>
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

TEST_CASE("SubsetSampler: Testing subset sampling strategies")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    GIVEN("A circular trajectory with 32 angles")
    {
        IndexVector_t size({{32, 32}});
        VolumeDescriptor volumeDescriptor{size};
        index_t numAngles{32}, arc{360};
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, volumeDescriptor, arc, static_cast<real_t>(size(0)) * 100.0f,
            static_cast<real_t>(size(0)));
        const auto numCoeffsPerDim = sinoDescriptor->getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = sinoDescriptor->getNumberOfDimensions();
        const auto numElements = numCoeffsPerDim[nDimensions - 1];

        WHEN("performing round robin sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            std::vector<index_t> indices(static_cast<std::size_t>(numElements));
            std::iota(indices.begin(), indices.end(), 0);
            const std::vector<std::vector<index_t>> mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRoundRobin(indices, nSubsets);
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
        WHEN("performing round robin sampling into 5 subsets")
        {
            const auto nSubsets = 5;
            std::vector<index_t> indices(static_cast<std::size_t>(numElements));
            std::iota(indices.begin(), indices.end(), 0);
            const auto mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRoundRobin(indices, nSubsets);
            THEN("The mapping is correct with every subset having 6 elements apart from the first "
                 "two")
            {
                for (std::size_t i = 0; i < static_cast<std::size_t>(nSubsets); ++i) {
                    if (i <= 1) {
                        REQUIRE_EQ(mapping[i].size(), 7);
                    } else {
                        REQUIRE_EQ(mapping[i].size(), 6);
                    }
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
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRotationalClustering(
                    static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                REQUIRE_EQ(mapping[0], std::vector<index_t>{15, 11, 7, 3, 0, 27, 23, 19});
                REQUIRE_EQ(mapping[1], std::vector<index_t>{14, 10, 6, 2, 30, 26, 22, 18});
                REQUIRE_EQ(mapping[2], std::vector<index_t>{13, 9, 5, 1, 29, 25, 21, 17});
                REQUIRE_EQ(mapping[3], std::vector<index_t>{12, 8, 4, 31, 28, 24, 20, 16});
            }
        }
    }

    GIVEN("A spherical trajectory with 32 angles and 4 circles")
    {
        IndexVector_t size({{32, 32, 32}});
        VolumeDescriptor volumeDescriptor{size};
        index_t numPoses{32}, numCircles{4};
        auto sinoDescriptor = SphereTrajectoryGenerator::createTrajectory(
            numPoses, volumeDescriptor, numCircles,
            geometry::SourceToCenterOfRotation(static_cast<real_t>(size(0)) * 100.0f),
            geometry::CenterOfRotationToDetector(static_cast<real_t>(size(0))));
        const auto numCoeffsPerDim = sinoDescriptor->getNumberOfCoefficientsPerDimension();
        const index_t nDimensions = sinoDescriptor->getNumberOfDimensions();
        const auto numElements = numCoeffsPerDim[nDimensions - 1];

        WHEN("performing round robin sampling into 4 subsets")
        {
            const auto nSubsets = 4;
            std::vector<index_t> indices(static_cast<std::size_t>(numElements));
            std::iota(indices.begin(), indices.end(), 0);
            const auto mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRoundRobin(indices, nSubsets);
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
        WHEN("performing round robin sampling into 5 subsets")
        {
            const auto nSubsets = 5;
            std::vector<index_t> indices(static_cast<std::size_t>(numElements));
            std::iota(indices.begin(), indices.end(), 0);
            const auto mapping =
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRoundRobin(indices, nSubsets);
            THEN("The mapping is correct with every subset having 6 elements apart from the first "
                 "two")
            {
                for (std::size_t i = 0; i < static_cast<std::size_t>(nSubsets); ++i) {
                    if (i <= 1) {
                        REQUIRE_EQ(mapping[i].size(), 7);
                    } else {
                        REQUIRE_EQ(mapping[i].size(), 6);
                    }
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
                SubsetSampler<PlanarDetectorDescriptor, real_t>::splitRotationalClustering(
                    static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets);
            THEN("The mapping is correct with every subset having 8 elements")
            {
                REQUIRE_EQ(mapping[0], std::vector<index_t>{0, 12, 1, 22, 30, 18, 25, 26});
                REQUIRE_EQ(mapping[1], std::vector<index_t>{4, 13, 21, 31, 23, 15, 28, 27});
                REQUIRE_EQ(mapping[2], std::vector<index_t>{5, 20, 10, 14, 24, 7, 17, 8});
                REQUIRE_EQ(mapping[3], std::vector<index_t>{11, 3, 6, 19, 29, 9, 16, 2});
            }
        }
    }
}

TEST_CASE(
    "SubsetSampler: 2D Integration test with PlanarDetectorDescriptor and circular trajectory")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    IndexVector_t size({{16, 16}});
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    index_t numAngles{20}, arc{360};
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
        WHEN("Setting up a subset sampler with 4 subsets and ROTATIONAL_CLUSTERING sampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets,
                SubsetSampler<PlanarDetectorDescriptor,
                              real_t>::SamplingStrategy::ROTATIONAL_CLUSTERING);
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

TEST_CASE(
    "SubsetSampler: 3D Integration test with PlanarDetectorDescriptor and spherical trajectory")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    IndexVector_t size({{8, 8, 8}});
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    index_t numPoses{16}, numCircles{5};
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

                    REQUIRE_EQ(coeffsPerDimension[0], coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                REQUIRE_EQ(subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>().size(),
                           nSubsets);
            }
        }
        WHEN("Setting up a subset sampler with 4 subsets and ROTATIONAL_CLUSTERING sampling")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), nSubsets,
                SubsetSampler<PlanarDetectorDescriptor,
                              real_t>::SamplingStrategy::ROTATIONAL_CLUSTERING);
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
