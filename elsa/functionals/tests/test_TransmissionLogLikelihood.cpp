/**
 * \file test_TransmissionLogLikelihood.cpp
 *
 * \brief Tests for the TransmissionLogLikelihood class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - rewrite
 */

#include <catch2/catch.hpp>
#include <cmath>
#include "TransmissionLogLikelihood.h"
#include "LinearResidual.h"
#include "Scaling.h"
#include "Identity.h"

using namespace elsa;

SCENARIO("Testing the TransmissionLogLikelihood functional") {
    GIVEN("just data (no residual)") {
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 13;
        DataDescriptor dd(numCoeff);

        RealVector_t y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer dcY(dd, y);

        RealVector_t b(dd.getNumberOfCoefficients());
        b.setRandom();
        // ensure b has positive values (due to log)
        for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
            if (b[i] < 0) b[i] *= -1;
            if (b[i] == 0) b[i] += 1;
        }
        DataContainer dcB(dd, b);

        WHEN("instantiating without r") {
            TransmissionLogLikelihood func(dd, dcY, dcB);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected") {
                auto tmllClone = func.clone();

                REQUIRE(tmllClone.get() != &func);
                REQUIRE(*tmllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = b[i] * std::exp(-dataVec[i]);
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = y[i] - temp;
                    trueScale[i] = temp;
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                DataContainer dcTrueScale(dd, trueScale);
                REQUIRE(func.getHessian(x) == leaf(Scaling(dd, dcTrueScale)));
            }
        }

        WHEN("instantiating with r") {
            RealVector_t r(dd.getNumberOfCoefficients());
            r.setRandom();
            // ensure non-negative values
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0) r[i] *= -1;
            DataContainer dcR(dd, r);

            TransmissionLogLikelihood func(dd, dcY, dcB, dcR);

            THEN("a clone behaves as expected") {
                auto tmllClone = func.clone();

                REQUIRE(tmllClone.get() != &func);
                REQUIRE(*tmllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the true values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = b[i] * std::exp(-dataVec[i]);
                    real_t tempR = temp + r[i];
                    trueValue += tempR - y[i] * std::log(tempR);
                    trueGrad[i] = (y[i] * temp) / tempR - temp;
                    trueScale[i] = temp + (r[i] * y[i] * temp) / (tempR * tempR);
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                DataContainer dcTrueScale(dd, trueScale);
                REQUIRE(func.getHessian(x) == leaf(Scaling(dd, dcTrueScale)));
            }
        }
    }

    GIVEN("a residual with data") {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 17;
        DataDescriptor dd(numCoeff);

        RealVector_t resData(dd.getNumberOfCoefficients());
        resData.setRandom();
        DataContainer dcResData(dd, resData);
        Identity idOp(dd);
        LinearResidual linRes(idOp, dcResData);

        RealVector_t y(dd.getNumberOfCoefficients());
        y.setRandom();
        DataContainer dcY(dd, y);

        RealVector_t b(dd.getNumberOfCoefficients());
        b.setRandom();
        // ensure b has positive values (due to log)
        for (index_t i = 0; i < dd.getNumberOfCoefficients(); ++i) {
            if (b[i] < 0) b[i] *= -1;
            if (b[i] == 0) b[i] += 1;
        }
        DataContainer dcB(dd, b);

        WHEN("instantiating without r") {
            TransmissionLogLikelihood func(linRes, dcY, dcB);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *lRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected") {
                auto tmllClone = func.clone();

                REQUIRE(tmllClone.get() != &func);
                REQUIRE(*tmllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = b[i] * std::exp(-dataVec[i] + resData[i]);
                    trueValue += temp - y[i] * std::log(temp);
                    trueGrad[i] = y[i] - temp;
                    trueScale[i] = temp;
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

        WHEN("instantiating with r") {
            RealVector_t r(dd.getNumberOfCoefficients());
            r.setRandom();
            // ensure non-negative values
            for (index_t i = 0; i < r.size(); ++i)
                if (r[i] < 0) r[i] *= -1;
            DataContainer dcR(dd, r);

            TransmissionLogLikelihood func(linRes, dcY, dcB, dcR);

            THEN("a clone behaves as expected") {
                auto tmllClone = func.clone();

                REQUIRE(tmllClone.get() != &func);
                REQUIRE(*tmllClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the true values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = b[i] * std::exp(-dataVec[i] + resData[i]);
                    real_t tempR = temp + r[i];
                    trueValue += tempR - y[i] * std::log(tempR);
                    trueGrad[i] = (y[i] * temp) / tempR - temp;
                    trueScale[i] = temp + (r[i] * y[i] * temp) / (tempR * tempR);
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