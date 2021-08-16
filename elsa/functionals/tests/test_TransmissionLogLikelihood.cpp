/**
 * @file test_TransmissionLogLikelihood.cpp
 *
 * @brief Tests for the TransmissionLogLikelihood class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - rewrite
 */

#include <doctest/doctest.h>

#include <cmath>
#include "testHelpers.h"
#include "TransmissionLogLikelihood.h"
#include "LinearResidual.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"
#include "TypeCasts.hpp"

using namespace elsa;
using namespace doctest;

TYPE_TO_STRING(std::complex<float>);
TYPE_TO_STRING(std::complex<double>);

TEST_SUITE_BEGIN("functionals");

TEST_CASE_TEMPLATE("TransmissionLogLikelihood: Testing with only data no residual", TestType, float,
                   double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 13;
        VolumeDescriptor dd(numCoeff);

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<TestType> dcY(dd, y);

        Vector b(dd.getNumberOfCoefficients());
        b.setRandom();
        // ensure b has positive values (due to log)
        for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
            if (b[i] < 0)
                b[i] *= -1;
            if (b[i] == 0)
                b[i] += 1;
        }
        DataContainer<TestType> dcB(dd, b);

        WHEN("instantiating without r")
        {
            TransmissionLogLikelihood func(dd, dcY, dcB);

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
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
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
                    auto temp = b[i] * std::exp(-dataVec[i]);
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = y[i] - temp;
                    trueScale[i] = temp;
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<TestType> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(func.getHessian(x), leaf(Scaling(dd, dcTrueScale)));
            }
        }

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            // ensure non-negative values
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<TestType> dcR(dd, r);

            TransmissionLogLikelihood func(dd, dcY, dcB, dcR);

            THEN("a clone behaves as expected")
            {
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                // compute the true values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = b[i] * std::exp(-dataVec[i]);
                    auto tempR = temp + r[i];
                    trueValue += tempR - y[i] * std::log(tempR);
                    trueGrad[i] = (y[i] * temp) / tempR - temp;
                    trueScale[i] = temp + (r[i] * y[i] * temp) / (tempR * tempR);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                DataContainer<TestType> dcTrueScale(dd, trueScale);
                REQUIRE_EQ(func.getHessian(x), leaf(Scaling(dd, dcTrueScale)));
            }
        }
    }
}

TEST_CASE_TEMPLATE("TransmissionLogLikelihood: Testing with residual", TestType, float, double)
{
    using Vector = Eigen::Matrix<TestType, Eigen::Dynamic, 1>;

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 17;
        VolumeDescriptor dd(numCoeff);

        Vector resData(dd.getNumberOfCoefficients());
        resData.setRandom();
        DataContainer<TestType> dcResData(dd, resData);
        Identity<TestType> idOp(dd);
        LinearResidual<TestType> linRes(idOp, dcResData);

        Vector y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer<TestType> dcY(dd, y);

        Vector b(dd.getNumberOfCoefficients());
        b.setRandom();
        // ensure b has positive values (due to log)
        for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
            if (b[i] < 0)
                b[i] *= -1;
            if (b[i] == 0)
                b[i] += 1;
        }
        DataContainer<TestType> dcB(dd, b);

        WHEN("instantiating without r")
        {
            TransmissionLogLikelihood func(linRes, dcY, dcB);

            THEN("the functional is as expected")
            {
                REQUIRE_EQ(func.getDomainDescriptor(), dd);

                auto* lRes = downcast_safe<LinearResidual<TestType>>(&func.getResidual());
                REQUIRE_UNARY(lRes);
                REQUIRE_EQ(*lRes, linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
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
                    auto temp = b[i] * std::exp(-dataVec[i] + resData[i]);
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = y[i] - temp;
                    trueScale[i] = temp;
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE_EQ(hx[i], dataVec[i] * trueScale[i]);
            }
        }

        WHEN("instantiating with r")
        {
            Vector r(dd.getNumberOfCoefficients());
            r.setRandom();
            // ensure non-negative values
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer<TestType> dcR(dd, r);

            TransmissionLogLikelihood func(linRes, dcY, dcB, dcR);

            THEN("a clone behaves as expected")
            {
                auto tmllClone = func.clone();

                REQUIRE_NE(tmllClone.get(), &func);
                REQUIRE_EQ(*tmllClone, func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                Vector dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer<TestType> x(dd, dataVec);

                // compute the true values
                TestType trueValue = 0;
                Vector trueGrad(dd.getNumberOfCoefficients());
                Vector trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    auto temp = b[i] * std::exp(-dataVec[i] + resData[i]);
                    auto tempR = temp + r[i];
                    trueValue += tempR - y[i] * std::log(tempR);
                    trueGrad[i] = (y[i] * temp) / tempR - temp;
                    trueScale[i] = temp + (r[i] * y[i] * temp) / (tempR * tempR);
                }

                REQUIRE_UNARY(checkApproxEq(func.evaluate(x), trueValue));
                DataContainer<TestType> dcTrueGrad(dd, trueGrad);
                REQUIRE_UNARY(isApprox(func.getGradient(x), dcTrueGrad));

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE_UNARY(checkApproxEq(hx[i], dataVec[i] * trueScale[i]));
            }
        }
    }
}

TEST_SUITE_END();
