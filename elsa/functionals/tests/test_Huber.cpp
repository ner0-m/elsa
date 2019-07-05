/**
 * \file test_Huber.cpp
 *
 * \brief Tests for the Huber class
 *
 * \author Matthias Wieczorek - initial code
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "Huber.h"
#include "LinearResidual.h"
#include "Identity.h"

using namespace elsa;

SCENARIO("Testing the Huber norm functional") {
    GIVEN("just data (no residual)") {
        IndexVector_t numCoeff(3);
        numCoeff << 7, 16, 29;
        DataDescriptor dd(numCoeff);

        real_t delta = 10;

        WHEN("instantiating") {
            Huber func(dd, delta);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);
            }

            THEN("a clone behaves as expected") {
                auto huberClone = func.clone();

                REQUIRE(huberClone.get() != &func);
                REQUIRE(*huberClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                // fix the first entries to be bigger/smaller than delta
                dataVec[0] = delta + 1; dataVec[1] = delta + 2; dataVec[2] = delta + 3;
                dataVec[3] = delta - 1; dataVec[4] = delta - 2; dataVec[5] = delta - 3;
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t value = dataVec[i];
                    if (std::abs(value) <= delta) {
                        trueValue += 0.5 * value * value;
                        trueGrad[i] = value;
                    } else {
                        trueValue += delta * (std::abs(value) - 0.5 * delta);
                        trueGrad[i] = (value > 0) ? delta : -delta;
                    }
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE(hx[i] == ((std::abs(dataVec[i]) <= delta) ? x[i] : 0));
            }
        }
    }

    GIVEN("a residual with data") {
        // linear residual
        IndexVector_t numCoeff(2);
        numCoeff << 47, 11;
        DataDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer b(dd, randomData);

        Identity A(dd);

        LinearResidual linRes(A, b);

        real_t delta = 20;

        WHEN("instantiating") {
            Huber func(linRes, delta);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *lRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected") {
                auto huberClone = func.clone();

                REQUIRE(huberClone.get() != &func);
                REQUIRE(*huberClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t value = dataVec[i] - randomData[i];
                    if (std::abs(value) <= delta) {
                        trueValue += 0.5 * value * value;
                        trueGrad[i] = value;
                    } else {
                        trueValue += delta * (std::abs(value) - 0.5 * delta);
                        trueGrad[i] = (value > 0) ? delta : -delta;
                    }
                }

                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer dcTrueGrad(dd, trueGrad);
                REQUIRE(func.getGradient(x) == dcTrueGrad);

                auto hessian = func.getHessian(x);
                auto hx = hessian.apply(x);
                for (index_t i = 0; i < hx.getSize(); ++i)
                    REQUIRE(hx[i] == ((std::abs(dataVec[i] - randomData[i]) <= delta) ? x[i] : 0));
            }
        }
    }
}