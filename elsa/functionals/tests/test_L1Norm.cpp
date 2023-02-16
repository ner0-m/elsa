#include <doctest/doctest.h>

#include "testHelpers.h"
#include "L1Norm.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(complex<float>);
TYPE_TO_STRING(complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("L1Norm: Testing without residual", TestType, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(1);
        numCoeff << 4;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            L1Norm<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);
            }

            THEN("a clone behaves as expected")
            {
                auto l1Clone = func.clone();

                REQUIRE_NE(l1Clone.get(), &func);
                REQUIRE_EQ(*l1Clone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec << -9, -4, 0, 1;
                DataContainer<TestType> dc(dd, dataVec);

                REQUIRE(checkApproxEq(func.evaluate(dc), 14));
                REQUIRE_THROWS_AS(func.getGradient(dc), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(dc), LogicError);
            }
        }
    }
}

TEST_SUITE_END();
