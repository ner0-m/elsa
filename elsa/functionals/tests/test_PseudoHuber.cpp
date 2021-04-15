/**
 * @file test_Pseudohuber.cpp
 *
 * @brief Tests for the Huber class
 *
 * @author Matthias Wieczorek - initial code
 * @author David Frank - rewrite
 * @author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "PseudoHuber.h"
#include "LinearResidual.h"
#include "LinearOperator.h"
#include "Scaling.h"
#include "Identity.h"
#include "VolumeDescriptor.h"

using namespace elsa;

SCENARIO("Testing the PseudoHuber norm functional")
{
    GIVEN("just data (no residual)")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 13, 34;
        VolumeDescriptor dd(numCoeff);

        real_t delta = 2;

        WHEN("instantiating")
        {
            PseudoHuber func(dd, delta);

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
                auto phClone = func.clone();

                REQUIRE(phClone.get() != &func);
                REQUIRE(*phClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = dataVec[i] / delta;
                    real_t tempSq = temp * temp;
                    real_t sqrtOnePTempSq = std::sqrt(static_cast<real_t>(1.0) + tempSq);
                    trueValue += delta * delta * (sqrtOnePTempSq - static_cast<real_t>(1.0));
                    trueGrad[i] = dataVec[i] / sqrtOnePTempSq;
                    trueScale[i] = (sqrtOnePTempSq - tempSq / sqrtOnePTempSq)
                                   / (static_cast<real_t>(1.0) + tempSq);
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
        // linear residual
        IndexVector_t numCoeff(3);
        numCoeff << 3, 7, 11;
        VolumeDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer b(dd, randomData);

        Identity A(dd);

        LinearResidual linRes(A, b);

        real_t delta = 1.5;

        WHEN("instantiating")
        {
            PseudoHuber func(linRes, delta);

            THEN("the functional is as expected")
            {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto* lRes = dynamic_cast<const LinearResidual<real_t>*>(&func.getResidual());
                REQUIRE(lRes);
                REQUIRE(*lRes == linRes);
            }

            THEN("a clone behaves as expected")
            {
                auto phClone = func.clone();

                REQUIRE(phClone.get() != &func);
                REQUIRE(*phClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected")
            {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                // compute the "true" values
                real_t trueValue = 0;
                RealVector_t trueGrad(dd.getNumberOfCoefficients());
                RealVector_t trueScale(dd.getNumberOfCoefficients());
                for (index_t i = 0; i < dataVec.size(); ++i) {
                    real_t temp = (dataVec[i] - randomData[i]) / delta;
                    real_t tempSq = temp * temp;
                    real_t sqrtOnePTempSq = std::sqrt(static_cast<real_t>(1.0) + tempSq);
                    trueValue += delta * delta * (sqrtOnePTempSq - static_cast<real_t>(1.0));
                    trueGrad[i] = (dataVec[i] - randomData[i]) / sqrtOnePTempSq;
                    trueScale[i] = (sqrtOnePTempSq - tempSq / sqrtOnePTempSq)
                                   / (static_cast<real_t>(1.0) + tempSq);
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