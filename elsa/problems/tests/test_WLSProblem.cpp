/**
 * \file test_WLSProblem.cpp
 *
 * \brief Tests for the WLSProblem class
 *
 * \author Matthias Wieczorek - initial code
 * \author Maximilian Hornung - modularization
 * \author David Frank - rewrite
 * \author Tobias Lasser - rewrite, modernization
 */

#include <catch2/catch.hpp>
#include "WLSProblem.h"
#include "Identity.h"
#include "Scaling.h"

using namespace elsa;

SCENARIO("Testing WLSProblem") {
    GIVEN("the operator and data") {
        IndexVector_t numCoeff(2); numCoeff << 7, 13;
        DataDescriptor dd(numCoeff);

        RealVector_t bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity idOp(dd);

        WHEN("setting up a ls problem without x0") {
            WLSProblem prob(idOp, dcB);

            THEN("the clone works correctly") {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected") {
                DataContainer dcZero(dd);
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * bVec.squaredNorm()));
                REQUIRE(prob.getGradient() == -1.0f * dcB);

                auto hessian = prob.getHessian();
                REQUIRE(hessian.apply(dcB) == dcB);
            }
        }

        WHEN("setting up a ls problem with x0") {
            RealVector_t x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer dcX0(dd, x0Vec);

            WLSProblem prob(idOp, dcB, dcX0);

            THEN("the clone works correctly") {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected") {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                REQUIRE(prob.evaluate() == Approx(0.5 * (x0Vec - bVec).squaredNorm()));
                REQUIRE(prob.getGradient() == (dcX0 - dcB));

                auto hessian = prob.getHessian();
                REQUIRE(hessian.apply(dcB) == dcB);
            }
        }
    }

    GIVEN("weights, operator and data") {
        IndexVector_t numCoeff(3); numCoeff << 7, 13, 17;
        DataDescriptor dd(numCoeff);

        RealVector_t bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer dcB(dd, bVec);

        Identity idOp(dd);

        RealVector_t weightsVec(dd.getNumberOfCoefficients());
        weightsVec.setRandom();
        DataContainer dcWeights(dd, weightsVec);
        Scaling scaleOp(dd, dcWeights);

        WHEN("setting up a wls problem without x0") {
            WLSProblem prob(scaleOp, idOp, dcB);

            THEN("the clone works correctly") {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected") {
                DataContainer dcZero(dd);
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * bVec.dot((weightsVec.array() * bVec.array()).matrix())));
                REQUIRE(prob.getGradient() == -1.0f * dcWeights * dcB);

                auto hessian = prob.getHessian();
                REQUIRE(hessian.apply(dcB) == dcWeights * dcB);
            }
        }

        WHEN("setting up a wls problem with x0") {
            RealVector_t x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer dcX0(dd, x0Vec);

            WLSProblem prob(scaleOp, idOp, dcB, dcX0);

            THEN("the clone works correctly") {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the problem behaves as expected") {
                DataContainer dcZero(dd);
                REQUIRE(prob.getCurrentSolution() == dcX0);

                REQUIRE(prob.evaluate() == Approx(0.5 * (x0Vec - bVec).dot((weightsVec.array() * (x0Vec - bVec).array()).matrix())));
                REQUIRE(prob.getGradient() == dcWeights * (dcX0 - dcB));

                auto hessian = prob.getHessian();
                REQUIRE(hessian.apply(dcB) == dcWeights * dcB);
            }
        }
    }
}