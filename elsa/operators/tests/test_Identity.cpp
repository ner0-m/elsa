/**
 * \file test_Identity.cpp
 *
 * \brief Tests for Identity class
 *
 * \author Tobias Lasser - main code
 */

#include <catch2/catch.hpp>
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Constructing an Identity operator")
{
    GIVEN("a descriptor")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 45, 28;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating an Identity operator")
        {
            Identity idOp(dd);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE(idOp.getDomainDescriptor() == dd);
                REQUIRE(idOp.getRangeDescriptor() == dd);
            }
        }

        WHEN("cloning an  Identity operator")
        {
            Identity idOp(dd);
            auto idOpClone = idOp.clone();

            THEN("everything matches")
            {
                REQUIRE(idOpClone.get() != &idOp);
                REQUIRE(*idOpClone == idOp);
            }
        }
    }
}

SCENARIO("Using Identity")
{
    GIVEN("some data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);
        DataContainer input(dd);
        input = 3.3f;

        Identity idOp(dd);

        WHEN("applying the identity")
        {
            auto output = idOp.apply(input);

            THEN("the result is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                REQUIRE(input == output);
            }
        }

        WHEN("applying the adjoint of identity")
        {
            auto output = idOp.applyAdjoint(input);

            THEN("the results is as expected")
            {
                REQUIRE(output.getSize() == input.getSize());
                REQUIRE(input == output);
            }
        }
    }
}
