#include "doctest/doctest.h"

#include "FBP.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "SiddonsMethod.h"
#include "CircleTrajectoryGenerator.h"
#include "Phantoms.h"
#include "TypeCasts.hpp"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE("SiddonsMethod Shepp-Logan Phantom Reconstruction")
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("a Phantom reconstruction problem")
    {

        IndexVector_t size({{128, 128}});
        auto phantom = phantoms::modifiedSheppLogan(size);
        auto& volumeDescriptor = phantom.getDataDescriptor();

        // generate circular trajectory
        index_t numAngles{200}, arc{360};
        const auto distance = static_cast<real_t>(size(0));
        auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            numAngles, phantom.getDataDescriptor(), arc, distance * 10000.0f, distance);

        // dynamic_cast to VolumeDescriptor is legal and will not throw, as Phantoms returns a
        // VolumeDescriptor
        SiddonsMethod projector(dynamic_cast<const VolumeDescriptor&>(volumeDescriptor),
                                *sinoDescriptor);

        // simulate the sinogram
        auto sinogram = projector.apply(phantom);

        auto ramlak = makeRamLak(sinogram.getDataDescriptor());
        auto shepplogan = makeSheppLogan(sinogram.getDataDescriptor());
        auto cosine = makeCosine(sinogram.getDataDescriptor());
        auto hann = makeHann(sinogram.getDataDescriptor());

        WHEN("setting up a FBP solver")
        {
            FBP fbp{projector, ramlak};

            THEN("applying it")
            {
                auto reconstruction = fbp.apply(sinogram);
                DataContainer resultsDifference = reconstruction - phantom;
                REQUIRE_LE(resultsDifference.l2Norm(),
                           epsilon * volumeDescriptor.getNumberOfCoefficients());
            }
        }
    }
}

TEST_SUITE_END();