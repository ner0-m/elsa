/**
 * @file test_ProjectionOperator.cpp
 *
 * @brief Tests for the ProjectionOperator class
 *
 * @author Andi Braimllari
 */

#include "ProjectionOperator.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>

using namespace elsa;

// TODO test when new logic is added to the ProjectionOperator class
SCENARIO("Constructing a Projection operator")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 45, 28;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating a Projection operator")
        {
            ProjectionOperator<real_t> projOp(dd);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE(projOp.getDomainDescriptor() == dd);
                REQUIRE(projOp.getRangeDescriptor() == dd);
            }
        }

        WHEN("cloning a Projection operator")
        {
            ProjectionOperator<real_t> projOp(dd);
            auto projOpClone = projOp.clone();

            THEN("everything matches")
            {
                REQUIRE(projOpClone.get() != &projOp);
                REQUIRE(*projOpClone == projOp);
            }
        }
    }
}

SCENARIO("Using Projection")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);
        DataContainer input(dd);
        input = 3.3f;

        ProjectionOperator<real_t> projOp(dd);

        WHEN("applying the projection")
        {
            auto output = projOp.apply(input);

            THEN("the result is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                REQUIRE(input == output);
            }
        }

        WHEN("applying the adjoint of projection")
        {
            auto output = projOp.applyAdjoint(input);

            THEN("the results is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                REQUIRE(input == output);
            }
        }
    }
}
