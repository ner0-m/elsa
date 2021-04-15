/**
 * @file test_SubsetSampler.cpp
 *
 * @brief Tests for the SubsetSampler class
 *
 * @author Michael Loipführer - initial code
 */
#include <catch2/catch.hpp>

#include "SubsetSampler.h"
#include "Logger.h"
#include "CircleTrajectoryGenerator.h"
#include "SiddonsMethod.h"
#include "PlanarDetectorDescriptor.h"
#include "PhantomGenerator.h"
#include "SiddonsMethod.h"
#include "JosephsMethod.h"

using namespace elsa;

SCENARIO("Testing SubsetSampler with PlanarDetectorDescriptor")
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
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), sinogram, nSubsets);
            THEN("the clone works as expected")
            {

                auto subsetSamplerClone = subsetSampler.clone();

                REQUIRE(subsetSamplerClone.get() != &subsetSampler);
                REQUIRE(*subsetSamplerClone == subsetSampler);
            }
        }
        WHEN("Setting up a subset sampler with 4 subsets")
        {
            index_t nSubsets{4};
            SubsetSampler<PlanarDetectorDescriptor, real_t> subsetSampler(
                static_cast<const VolumeDescriptor&>(volumeDescriptor),
                static_cast<const PlanarDetectorDescriptor&>(*sinoDescriptor), sinogram, nSubsets);

            THEN("The full data has the correct dimensions")
            {
                const auto data = subsetSampler.getData();
                REQUIRE(data.getDataDescriptor().getNumberOfDimensions()
                        == sinogram.getDataDescriptor().getNumberOfDimensions());
                REQUIRE(data.getDataDescriptor().getNumberOfCoefficientsPerDimension()
                        == coeffsPerDimSinogram);
            }

            THEN("It returns the correct data blocks and projectors for each subsets")
            {
                for (index_t i = 0; i < nSubsets; i++) {
                    const auto subset = subsetSampler.getData().getBlock(i);
                    const auto coeffsPerDimension =
                        subset.getDataDescriptor().getNumberOfCoefficientsPerDimension();

                    REQUIRE(coeffsPerDimension[0] == coeffsPerDimSinogram[0]);
                }

                subsetSampler.getProjector<SiddonsMethod<real_t>>();
                subsetSampler.getSubsetProjectors<SiddonsMethod<real_t>>();
            }
        }
    }
}