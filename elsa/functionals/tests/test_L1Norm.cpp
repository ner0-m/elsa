/**
 * @file test_L1Norm.cpp
 *
 * @brief Tests for the L1Norm class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "L1Norm.h"
#include "LinearResidual.h"
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
        VolumeDescriptor dd({4});

        WHEN("instantiating")
        {
            L1Norm<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto& residual = func.getResidual();
                auto* linRes = downcast_safe<LinearResidual<TestType>>(&residual);
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
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

TEST_CASE_TEMPLATE("L1Norm: Testing with residual", TestType, float, double, complex<float>,
                   complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a residual with data")
    {
        VolumeDescriptor dd({4});

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(dd, randomData);

        Identity<TestType> idOp(dd);

        LinearResidual<TestType> linRes(idOp, dc);

        WHEN("instantiating")
        {
            L1Norm<TestType> func(linRes);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto& residual = func.getResidual();
                auto* lRes = downcast_safe<LinearResidual<TestType>>(&residual);
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
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
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                REQUIRE_UNARY(
                    checkApproxEq(func.evaluate(x), (dataVec - randomData).template lpNorm<1>()));
                REQUIRE_THROWS_AS(func.getGradient(x), LogicError);
                REQUIRE_THROWS_AS(func.getHessian(x), LogicError);
            }
        }
    }
}

TEST_SUITE_END();
