/**
 * @file test_EmissionLogLikelihood.cpp
 *
 * @brief Tests for the EmissionLogLikelihood class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - rewrite
 */

#include <catch2/catch.hpp>
#include <cmath>
#include "EmissionLogLikelihood.h"
#include "LinearResidual.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Testing the EmissionLogLikelihood functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 9, 15, 19;
        VolumeDescriptor dd(numCoeff);

        RealVector_t y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer dcY(dd, y);

        WHEN("instantiating without r")
        {
            EmissionLogLikelihood func(dd, dcY);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto* linRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE(emllClone.get() != &func);
                REQUIRE(*emllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = dataVec[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                DataContainer dcTrueScale(dd, trueScale);
                REQUIRE(func.getHessian(x) == leaf(Scaling(dd, dcTrueScale)));
            }
        }

        WHEN("instantiating with r")
        {
            RealVector_t r(dd.getNumberOfCoefficients());
            r.setRandom();
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer dcR(dd, r);

            EmissionLogLikelihood func(dd, dcY, dcR);

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE(emllClone.get() != &func);
                REQUIRE(*emllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                for (index_t i = 0; i < dataVec.size(); ++i) // ensure non-negative numbers
                    if (dataVec[i] < 0)
                        dataVec[i] *= -1;
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = dataVec[i] + r[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                DataContainer dcTrueScale(dd, trueScale);
                REQUIRE(func.getHessian(x) == leaf(Scaling(dd, dcTrueScale)));
            }
        }
    }

    GIVEN("a residual with data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 15, 21;
        VolumeDescriptor dd(numCoeff);

        RealVector_t resData(dd.getNumberOfCoefficients());
        resData.setRandom();
        DataContainer dcResData(dd, resData);
        Identity idOp(dd);
        LinearResidual linRes(idOp, dcResData);

        RealVector_t y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer dcY(dd, y);

        WHEN("instantiating without r")
        {
            EmissionLogLikelihood func(linRes, dcY);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE(emllClone.get() != &func);
                REQUIRE(*emllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                // ensure non-negative numbers
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    if (dataVec[i] - resData[i] < 0)
                        dataVec[i] -= 2 * (dataVec[i] - resData[i]);
                }
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = dataVec[i] - resData[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE(hx[i] == Approx(dataVec[i] * trueScale[i]));
            }
        }

        WHEN("instantiating with r")
        {
            RealVector_t r(dd.getNumberOfCoefficients());
            r.setRandom();
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0)
                    r[i] *= -1;
            DataContainer dcR(dd, r);

            EmissionLogLikelihood func(linRes, dcY, dcR);

            THEN("a clone behaves as expected")
            {
                auto emllClone = func.clone();

                REQUIRE(emllClone.get() != &func);
                REQUIRE(*emllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                // ensure non-negative numbers
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    if (dataVec[i] - resData[i] < 0)
                        dataVec[i] -= 2 * (dataVec[i] - resData[i]);
                }
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = dataVec[i] - resData[i] + r[i];
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = 1 - y[i] / temp;
                    trueScale[i] = y[i] / (temp * temp);
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE(hx[i] == Approx(dataVec[i] * trueScale[i]));
            }
        }
    }
}