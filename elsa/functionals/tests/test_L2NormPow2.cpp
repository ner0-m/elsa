/**
 * @file test_L2NormPow2.cpp
 *
 * @brief Tests for the L2NormPow2 class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "L2NormPow2.h"
#include "LinearResidual.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(std::complex<float>);
TYPE_TO_STRING(std::complex<double>);

TEST_SUITE_BEGIN("functionals");

// SCENARIO("Testing the l2 norm (squared) functional")
TEST_CASE_TEMPLATE("WeightedL2NormPow2: Testing without residual", TestType, float, double,
                   std::complex<float>, std::complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 11, 13;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            L2NormPow2<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
            }

            THEN("a clone behaves as expected")
            {
                auto l2Clone = func.clone();
                ;

                REQUIRE_NE(l2Clone.get(), &func);
                REQUIRE_EQ(*l2Clone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), 0.5f * dataVec.squaredNorm()));
                REQUIRE_EQ(func.getGradient(x), x);

                Identity<TestType> idOp(dd);
                REQUIRE_EQ(func.getHessian(x), leaf(idOp));
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> b(dd, randomData);

        Identity<TestType> A(dd);

        LinearResidual<TestType> linRes(A, b);

        WHEN("instantiating")
        {
            L2NormPow2<TestType> func(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto l2Clone = func.clone();

                REQUIRE_NE(l2Clone.get(), &func);
                REQUIRE_EQ(*l2Clone, func);
            }

            THEN("the evaluate, gradient and Hessian work was expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                REQUIRE_UNARY(
                    checkApproxEq(func.evaluate(x), 0.5f * (dataVec - randomData).squaredNorm()));

                DataContainer<TestType> grad(dd, (dataVec - randomData).eval());
                REQUIRE_EQ(func.getGradient(x), grad);

                auto hessian = func.getHessian(x);
                REQUIRE_EQ(hessian.apply(x), x);
            }
        }
    }
}

TEST_SUITE_END();
