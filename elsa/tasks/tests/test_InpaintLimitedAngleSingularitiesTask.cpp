/**
 * @file test_InpaintLimitedAngleSingularitiesTask.cpp
 *
 * @brief Tests for the InpaintLimitedAngleSingularitiesTask class
 *
 * @author Andi Braimllari
 */

#include "InpaintLimitedAngleSingularitiesTask.h"
#include "VolumeDescriptor.h"
#include "doctest/doctest.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("tasks");

TEST_CASE_TEMPLATE("InpaintLimitedAngleSingularitiesTask: Doing stuff", TestType, float, double)
{
    GIVEN("some non-square images and differently-sized vectors of DataContainers")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 256, 256;
        VolumeDescriptor volDescr(numCoeff);

        IndexVector_t anotherNumCoeff(2);
        anotherNumCoeff << 256, 512;
        VolumeDescriptor anotherVolDescr(anotherNumCoeff);

        IndexVector_t threeDimNumCoeff(3);
        threeDimNumCoeff << 128, 256, 512;
        VolumeDescriptor threeDimVolDescr(threeDimNumCoeff);

        WHEN("instantiating a InpaintLimitedAngleSingularitiesTask object")
        {
            InpaintLimitedAngleSingularitiesTask<TestType> laTask;
            THEN("reconstructVisibleCoeffsOfLimitedAngleCT throws for  works as intended")
            {
                DataContainer<TestType> threeDimImage(threeDimVolDescr);
                std::pair<elsa::geometry::Degree, elsa::geometry::Degree> missingWedgeAngles(
                    elsa::geometry::Degree(40), elsa::geometry::Degree(80));

                REQUIRE_THROWS_AS(laTask.reconstructVisibleCoeffsOfLimitedAngleCT(
                                      threeDimImage, missingWedgeAngles),
                                  InvalidArgumentError);
            }

            THEN("trainPhantomNet throws for different count of inputs and labels")
            {
                std::vector<DataContainer<TestType>> inputs;
                DataContainer<TestType> noise(volDescr);
                inputs.emplace_back(noise);
                std::vector<DataContainer<TestType>> labels;

                REQUIRE_THROWS_AS(laTask.trainPhantomNet(inputs, labels), LogicError);
            }

            THEN("combineVisCoeffsToInpaintedInvisCoeffs throws for different DataDescriptors of "
                 "the visCoeffs and invisCoeffs, as well as for non-squared images")
            {
                DataContainer<TestType> visCoeffs(volDescr);
                DataContainer<TestType> invisCoeffs(anotherVolDescr);

                REQUIRE_THROWS_AS(
                    laTask.combineVisCoeffsToInpaintedInvisCoeffs(visCoeffs, invisCoeffs),
                    LogicError);
                REQUIRE_THROWS_AS(
                    laTask.combineVisCoeffsToInpaintedInvisCoeffs(visCoeffs, invisCoeffs),
                    InvalidArgumentError);
            }
        }
    }
}

TEST_SUITE_END();
