/**
 * @file test_EmissionLogLikelihood.cpp
 *
 * @brief Tests for the EmissionLogLikelihood class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - rewrite
 */

#include <doctest/doctest.h>

#include <cmath>
#include "testHelpers.h"
#include "EmissionLogLikelihood.h"
#include "LinearResidual.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("EmissionLogLikelihood: Testing with only data no residual", TestType, float,
                   double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 9, 15, 19;
        VolumeDescriptor dd(numCoeff);

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<TestType> dcY(dd, y);

        WHEN("instantiating without r")
        {
            EmissionLogLikelihood func(dd, dcY);

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
                auto emllClone = func.clone();

                REQUIRE_NE(emllClone.get(), &func);
                REQUIRE_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    TestType temp = dataVec[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<TestType> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(*func.getHessian(x), Scaling(dd, dcTrueScale));
            }
        }

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<TestType> dcR(dd, r);

            EmissionLogLikelihood func(dd, dcY, dcR);

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE_NE(emllClone.get(), &func);
                REQUIRE_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = dataVec[i] + r[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
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

TEST_CASE_TEMPLATE("EmissionLogLikelihood: Testing with residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 15, 21;
        VolumeDescriptor dd(numCoeff);

        Vector resData(dd.getNumberOfCoefficients());
        resData.setRandom();
        DataContainer<TestType> dcResData(dd, resData);
        Identity<TestType> idOp(dd);
        LinearResidual<TestType> linRes(idOp, dcResData);

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<TestType> dcY(dd, y);

        WHEN("instantiating without r")
        {
            EmissionLogLikelihood func(linRes, dcY);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE_NE(emllClone.get(), &func);
                REQUIRE_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                // ensure non-negative numbers
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    if (dataVec[i] - resData[i] < 0)
                        dataVec[i] -= 2 * (dataVec[i] - resData[i]);
                }
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = dataVec[i] - resData[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
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

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<TestType> dcR(dd, r);

            EmissionLogLikelihood func(linRes, dcY, dcR);

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE_NE(emllClone.get(), &func);
                REQUIRE_EQ(*emllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                // ensure non-negative numbers
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    if (dataVec[i] - resData[i] < 0)
                        dataVec[i] -= 2 * (dataVec[i] - resData[i]);
                }
                DataContainer<TestType> x(dd, dataVec);

                // compute the "true" values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = dataVec[i] - resData[i] + r[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                auto hessian = func.getHessian(x);
                auto hx = hessian->apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE_EQ(hx[i], dataVec[i] * trueScale[i]);
            }
        }
    }
}

TEST_SUITE_END();
