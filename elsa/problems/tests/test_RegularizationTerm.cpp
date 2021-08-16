#include "doctest/doctest.h"

#include "RegularizationTerm.h"
#include "L2NormPow2.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE("RegularizationTerm: Test a simple term")
{
    GIVEN("some term")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 16;
        VolumeDescriptor dd(numCoeff);

        real_t weight = 2.5;
        L2NormPow2 functional(dd);

        WHEN("instantiating")
        {
            RegularizationTerm regTerm(weight, functional);

            THEN("everything is as expected")
            {
                REQUIRE_EQ(regTerm.getWeight(), Approx(weight));
                REQUIRE_EQ(regTerm.getFunctional(), functional);
            }
        }
    }

    GIVEN("another regularization term")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 11, 17, 23;
        VolumeDescriptor dd(numCoeff);

        auto weight = real_t{3.1f};
        L2NormPow2 functional(dd);

        RegularizationTerm regTerm(weight, functional);

        WHEN("copy constructing")
        {
            RegularizationTerm rt(regTerm);

            THEN("it copied correctly")
            {
                REQUIRE_EQ(rt.getWeight(), Approx(regTerm.getWeight()));
                REQUIRE_EQ(rt.getFunctional(), regTerm.getFunctional());

                REQUIRE_EQ(rt, regTerm);
            }
        }

        WHEN("copy assigning")
        {
            RegularizationTerm rt(0.0f, functional);
            rt = regTerm;

            THEN("it copied correctly")
            {
                REQUIRE_EQ(rt.getWeight(), Approx(regTerm.getWeight()));
                REQUIRE_EQ(rt.getFunctional(), regTerm.getFunctional());

                REQUIRE_EQ(rt, regTerm);
            }
        }

        WHEN("move constructing")
        {
            RegularizationTerm oldOtherRt(regTerm);

            RegularizationTerm rt(std::move(regTerm));

            THEN("it moved correctly")
            {
                REQUIRE_EQ(rt.getWeight(), Approx(oldOtherRt.getWeight()));
                REQUIRE_EQ(rt.getFunctional(), oldOtherRt.getFunctional());

                REQUIRE_EQ(rt, oldOtherRt);
            }
        }

        WHEN("move assigning")
        {
            RegularizationTerm oldOtherRt(regTerm);

            RegularizationTerm rt(0.0f, functional);
            rt = std::move(regTerm);

            THEN("it moved correctly")
            {
                REQUIRE_EQ(rt.getWeight(), Approx(oldOtherRt.getWeight()));
                REQUIRE_EQ(rt.getFunctional(), oldOtherRt.getFunctional());

                REQUIRE_EQ(rt, oldOtherRt);
            }
        }
    }
}

TEST_SUITE_END();
