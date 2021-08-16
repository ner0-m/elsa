/**
 * @file test_Constraint.cpp
 *
 * @brief Tests for the Constraint class
 *
 * @author Andi Braimllari
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "Constraint.h"
#include "Identity.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(std::complex<float>);
TYPE_TO_STRING(std::complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("Constraint: Testing construction and clone", TestType, float,
                   std::complex<float>, double, std::complex<double>)
{
    GIVEN("an Identity, a Scaling operator and a DataContainer")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 24;
        VolumeDescriptor dd(numCoeff);

        Identity<TestType> A(dd);
        Scaling<TestType> B(dd, -1);
        DataContainer<TestType> c(dd);
        c = 0;

        WHEN("instantiating")
        {
            Constraint<TestType> constraint(A, B, c);

            THEN("the Constraint is as expected")
            {
                REQUIRE_EQ(constraint.getOperatorA(), A);
                REQUIRE_EQ(constraint.getOperatorB(), B);
                REQUIRE_UNARY(isApprox(constraint.getDataVectorC(), c));
            }

            THEN("a clone behaves as expected")
            {
                auto constraintClone = constraint.clone();

                REQUIRE_NE(constraintClone.get(), &constraint);
                REQUIRE_EQ(*constraintClone, constraint);
            }
        }
    }
}

TEST_SUITE_END();
