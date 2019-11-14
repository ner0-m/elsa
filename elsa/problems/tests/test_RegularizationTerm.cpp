/**
 *
 */

#include <catch2/catch.hpp>
#include "RegularizationTerm.h"
#include "L2NormPow2.h"

using namespace elsa;

SCENARIO("Testing RegularizationTerm")
{
    GIVEN("some term")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 16;
        DataDescriptor dd(numCoeff);

        real_t weight = 2.5;
        L2NormPow2 functional(dd);

        WHEN("instantiating")
        {
            RegularizationTerm regTerm(weight, functional);

            THEN("everything is as expected")
            {
                REQUIRE(regTerm.getWeight() == weight);
                REQUIRE(regTerm.getFunctional() == functional);
            }
        }
    }

    GIVEN("another regularization term")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 17, 23;
        DataDescriptor dd(numCoeff);

        real_t weight = 3.1;
        L2NormPow2 functional(dd);

        RegularizationTerm regTerm(weight, functional);

        WHEN("copy constructing")
        {
            RegularizationTerm rt(regTerm);

            THEN("it copied correctly")
            {
                REQUIRE(rt.getWeight() == regTerm.getWeight());
                REQUIRE(rt.getFunctional() == regTerm.getFunctional());

                REQUIRE(rt == regTerm);
            }
        }

        WHEN("copy assigning")
        {
            RegularizationTerm rt(0.0f, functional);
            rt = regTerm;

            THEN("it copied correctly")
            {
                REQUIRE(rt.getWeight() == regTerm.getWeight());
                REQUIRE(rt.getFunctional() == regTerm.getFunctional());

                REQUIRE(rt == regTerm);
            }
        }

        WHEN("move constructing")
        {
            RegularizationTerm oldOtherRt(regTerm);

            RegularizationTerm rt(std::move(regTerm));

            THEN("it moved correctly")
            {
                REQUIRE(rt.getWeight() == oldOtherRt.getWeight());
                REQUIRE(rt.getFunctional() == oldOtherRt.getFunctional());

                REQUIRE(rt == oldOtherRt);
            }
        }

        WHEN("move assigning")
        {
            RegularizationTerm oldOtherRt(regTerm);

            RegularizationTerm rt(0.0f, functional);
            rt = std::move(regTerm);

            THEN("it moved correctly")
            {
                REQUIRE(rt.getWeight() == oldOtherRt.getWeight());
                REQUIRE(rt.getFunctional() == oldOtherRt.getFunctional());

                REQUIRE(rt == oldOtherRt);
            }
        }
    }
}