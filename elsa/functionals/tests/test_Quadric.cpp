/**
 * @file test_Quadric.cpp
 *
 * @brief Tests for the Quadric class
 *
 * @author Matthias Wieczorek - initial code
 * @author Maximilian Hornung - modularization
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "Quadric.h"
#include "Identity.h"
#include "Scaling.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(std::complex<float>);
TYPE_TO_STRING(std::complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("Quadric: Testing without residual", TestType, float, double,
                   std::complex<float>, std::complex<double>)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("no operator and no data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd(numCoeff);

        WHEN("instantiating")
        {
            Quadric<TestType> func(dd);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE_UNARY_FALSE(gradExpr.hasDataVector());
                REQUIRE_UNARY_FALSE(gradExpr.hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto qClone = func.clone();

                REQUIRE_NE(qClone.get(), &func);
                REQUIRE_EQ(*qClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                TestType trueValue = static_cast<TestType>(0.5) * x.squaredL2Norm();
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                REQUIRE_UNARY(isApprox(func.getGradient(x), x));
                REQUIRE_EQ(func.getHessian(x), leaf(Identity<TestType>(dd)));
            }
        }
    }

    GIVEN("an operator but no data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd(numCoeff);

        Scaling scalingOp(dd, static_cast<TestType>(3.0));

        WHEN("instantiating")
        {
            Quadric<TestType> func(scalingOp);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE_UNARY_FALSE(gradExpr.hasDataVector());
                REQUIRE_EQ(gradExpr.getOperator(), scalingOp);
            }

            THEN("a clone behaves as expected")
            {
                auto qClone = func.clone();

                REQUIRE_NE(qClone.get(), &func);
                REQUIRE_EQ(*qClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                TestType trueValue =
                    static_cast<TestType>(0.5) * scalingOp.getScaleFactor() * x.squaredL2Norm();
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                REQUIRE_EQ(func.getGradient(x), scalingOp.getScaleFactor() * x);
                REQUIRE_EQ(func.getHessian(x), leaf(scalingOp));
            }
        }
    }

    GIVEN("data but no operator")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(dd, randomData);

        WHEN("instantiating")
        {
            Quadric<TestType> func(dc);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE_EQ(gradExpr.getDataVector(), dc);
                REQUIRE_UNARY_FALSE(gradExpr.hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto qClone = func.clone();

                REQUIRE_NE(qClone.get(), &func);
                REQUIRE_EQ(*qClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                TestType trueValue = static_cast<TestType>(0.5) * x.squaredL2Norm() - x.dot(dc);
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                REQUIRE_EQ(func.getGradient(x), x - dc);
                REQUIRE_EQ(func.getHessian(x), leaf(Identity<TestType>(dd)));
            }
        }
    }

    GIVEN("an operator and data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 13, 11, 7;
        VolumeDescriptor dd(numCoeff);

        Identity<TestType> idOp(dd);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> dc(dd, randomData);

        WHEN("instantiating")
        {
            Quadric<TestType> func(idOp, dc);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE(isApprox(gradExpr.getDataVector(), dc));
                REQUIRE_EQ(gradExpr.getOperator(), idOp);
            }

            THEN("a clone behaves as expected")
            {
                auto qClone = func.clone();

                REQUIRE_NE(qClone.get(), &func);
                REQUIRE_EQ(*qClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                TestType trueValue = static_cast<TestType>(0.5) * x.dot(x) - x.dot(dc);
                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> grad(dd, (dataVec - randomData).eval());
                REQUIRE_EQ(func.getGradient(x), grad);
                REQUIRE_EQ(func.getHessian(x), leaf(idOp));
            }
        }
    }
}

TEST_SUITE_END();
