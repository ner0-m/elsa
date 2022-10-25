/**
 * @file test_Pseudohuber.cpp
 *
 * @brief Tests for the Huber class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <doctest/doctest.h>

#include "testHelpers.h"
#include "PseudoHuber.h"
#include "LinearResidual.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("PseudoHuber: Testing without residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 34;
        VolumeDescriptor dd(numCoeff);

        real_t delta = 2;

        WHEN("instantiating")
        {
            PseudoHuber<TestType> func(dd, delta);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* linRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(linRes);
                REQUIRE_UNARY_FALSE(linRes->hasDataVector());
                REQUIRE_UNARY_FALSE(linRes->hasOperator());
            }

            THEN("a clone behaves as expected")
            {
                auto phClone = func.clone();

                REQUIRE_NE(phClone.get(), &func);
                REQUIRE_EQ(*phClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    TestType temp = dataVec[i] / delta;
                    TestType tempSq = temp * temp;
                    TestType sqrtOnePTempSq = std::sqrt(static_cast<TestType>(1.0) + tempSq);
                    trueValue += delta * delta * (sqrtOnePTempSq - static_cast<TestType>(1.0));
                    trueGrad[i] = dataVec[i] / sqrtOnePTempSq;
                    trueScale[i] = (sqrtOnePTempSq - tempSq / sqrtOnePTempSq)
                                   / (static_cast<TestType>(1.0) + tempSq);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<TestType> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(*func.getHessian(x), Scaling(dd, dcTrueScale));
            }
        }
    }
}

TEST_CASE_TEMPLATE("PseudoHuber: Testing with residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a residual with data")
    {
        // linear residual
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 11;
        VolumeDescriptor dd(numCoeff);

        Vector randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer<TestType> b(dd, randomData);

        Identity<TestType> A(dd);

        LinearResidual<TestType> linRes(A, b);

        real_t delta = static_cast<real_t>(1.5);

        WHEN("instantiating")
        {
            PseudoHuber<TestType> func(linRes, delta);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto phClone = func.clone();

                REQUIRE_NE(phClone.get(), &func);
                REQUIRE_EQ(*phClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    TestType temp = (dataVec[i] - randomData[i]) / delta;
                    TestType tempSq = temp * temp;
                    TestType sqrtOnePTempSq = std::sqrt(static_cast<TestType>(1.0) + tempSq);
                    trueValue += delta * delta * (sqrtOnePTempSq - static_cast<TestType>(1.0));
                    trueGrad[i] = (dataVec[i] - randomData[i]) / sqrtOnePTempSq;
                    trueScale[i] = (sqrtOnePTempSq - tempSq / sqrtOnePTempSq)
                                   / (static_cast<TestType>(1.0) + tempSq);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                auto hessian = func.getHessian(x);
                auto hx = hessian->apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(hx[i], dataVec[i] * trueScale[i]));
            }
        }
    }
}

TEST_SUITE_END();
