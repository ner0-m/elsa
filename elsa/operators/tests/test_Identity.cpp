/**
 * @file test_Identity.cpp
 *
 * @brief Tests for Identity class
 *
 * @author Tobias Lasser - main code
 */

#include "doctest/doctest.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("core");

TEST_CASE_TEMPLATE("Identity: Testing construction", data_t, float, double)
{
    GIVEN("a descriptor")
    {
        VolumeDescriptor dd({13, 45, 28});

        WHEN("instantiating an Identity operator")
        {
            Identity<data_t> idOp(dd);

            THEN("the DataDescriptors are as expected")
            {
                REQUIRE(idOp.getDomainDescriptor() == dd);
                REQUIRE(idOp.getRangeDescriptor() == dd);
            }
        }

        WHEN("cloning an  Identity operator")
        {
            Identity<data_t> idOp(dd);
            auto idOpClone = idOp.clone();

            THEN("everything matches")
            {
                REQUIRE(idOpClone.get() != &idOp);
                REQUIRE(*idOpClone == idOp);
            }
        }
    }
}

TEST_CASE_TEMPLATE("Identity: Testing apply", data_t, float, double, complex<float>,
                   complex<double>)
{
    GIVEN("some data")
    {
        VolumeDescriptor dd({11, 13});
        DataContainer<data_t> input(dd);
        input = 3.3f;

        Identity<data_t> idOp(dd);

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
TEST_SUITE_END();
