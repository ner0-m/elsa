#include "Constraint.h"
#include "Identity.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"

#include <catch2/catch.hpp>

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing Constraint", "", float, std::complex<float>, double,
                   std::complex<double>)
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
                REQUIRE(constraint.getOperatorA() == A);
                REQUIRE(constraint.getOperatorB() == B);
                REQUIRE(constraint.getDataVectorC() == c);
            }

            THEN("a clone behaves as expected")
            {
                auto constraintClone = constraint.clone();

                REQUIRE(constraintClone.get() != &constraint);
                REQUIRE(*constraintClone == constraint);
            }
        }
    }
}
