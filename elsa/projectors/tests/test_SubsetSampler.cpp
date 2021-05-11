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

SCENARIO("Testing SubsetSampler with PlanarDetectorDescriptor and circular trajectory")
{
    Logger::setLevel(Logger::LogLevel::WARN);

    IndexVector_t size(2);
    size << 128, 128;
    auto phantom = PhantomGenerator<real_t>::createModifiedSheppLogan(size);
    auto& volumeDescriptor = phantom.getDataDescriptor();

    index_t numAngles{180}, arc{360};
    auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
        numAngles, phantom.getDataDescriptor(), arc, static_cast<real_t>(size(0)) * 100,
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
        numPoses, phantom.getDataDescriptor(), numCircles, static_cast<real_t>(size(0)) * 100,
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
