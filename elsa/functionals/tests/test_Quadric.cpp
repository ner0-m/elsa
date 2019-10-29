/**
 * \file test_Quadric.cpp
 *
 * \brief Tests for the Quadric class
 *
 * \author Matthias Wieczorek - initial code
 * \author Maximilian Hornung - modularization
 * \author David Frank - rewrite
 * \author Tobias Lasser - modernization
 */

#include <catch2/catch.hpp>
#include "Quadric.h"
#include "Identity.h"
#include "Scaling.h"

using namespace elsa;

SCENARIO("Testing the Quadric functional") {
    GIVEN("no operator and no data") {
        IndexVector_t numCoeff(3); numCoeff << 13, 11, 7;
        DataDescriptor dd(numCoeff);

        WHEN("instantiating") {
            Quadric func(dd);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE(gradExpr.hasDataVector() == false);
                REQUIRE(gradExpr.hasOperator() == false);
            }

            THEN("a clone behaves as expected") {
                auto qClone = func.clone();

                REQUIRE(qClone.get() != &func);
                REQUIRE(*qClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                real_t trueValue = static_cast<real_t>(0.5)*x.squaredL2Norm();
                REQUIRE(func.evaluate(x) == Approx(trueValue));
                REQUIRE(func.getGradient(x) == x);
                REQUIRE(func.getHessian(x) == leaf(Identity(dd)));
            }
        }
    }

    GIVEN("an operator but no data") {
        IndexVector_t numCoeff(3); numCoeff << 13, 11, 7;
        DataDescriptor dd(numCoeff);

        Scaling scalingOp(dd,static_cast<real_t>(3.0));

        WHEN("instantiating") {
            Quadric func(scalingOp);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE(gradExpr.hasDataVector() == false);
                REQUIRE(gradExpr.getOperator() == scalingOp);
            }

            THEN("a clone behaves as expected") {
                auto qClone = func.clone();

                REQUIRE(qClone.get() != &func);
                REQUIRE(*qClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                real_t trueValue = static_cast<real_t>(0.5)*scalingOp.getScaleFactor()*x.squaredL2Norm();
                REQUIRE(func.evaluate(x) == Approx(trueValue));
                REQUIRE(func.getGradient(x) == scalingOp.getScaleFactor()*x);
                REQUIRE(func.getHessian(x) == leaf(scalingOp));
            }
        }
    }

    GIVEN("data but no operator") {
        IndexVector_t numCoeff(3); numCoeff << 13, 11, 7;
        DataDescriptor dd(numCoeff);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(dd, randomData);

        WHEN("instantiating") {
            Quadric func(dc);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE(gradExpr.getDataVector() == dc);
                REQUIRE(gradExpr.hasOperator() == false);
            }

            THEN("a clone behaves as expected") {
                auto qClone = func.clone();

                REQUIRE(qClone.get() != &func);
                REQUIRE(*qClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                real_t trueValue = static_cast<real_t>(0.5)*x.squaredL2Norm() - x.dot(dc);
                REQUIRE(func.evaluate(x) == Approx(trueValue));
                REQUIRE(func.getGradient(x) == x - dc);
                REQUIRE(func.getHessian(x) == leaf(Identity<real_t>(dd)));
            }
        }
    } 

    GIVEN("an operator and data") {
        IndexVector_t numCoeff(3); numCoeff << 13, 11, 7;
        DataDescriptor dd(numCoeff);

        Identity idOp(dd);

        RealVector_t randomData(dd.getNumberOfCoefficients());
        randomData.setRandom();
        DataContainer dc(dd, randomData);

        WHEN("instantiating") {
            Quadric func(idOp, dc);

            THEN("the functional is as expected") {
                REQUIRE(func.getDomainDescriptor() == dd);

                auto *linRes = dynamic_cast<const LinearResidual<real_t> *>(&func.getResidual());
                REQUIRE(linRes);
                REQUIRE(linRes->hasDataVector() == false);
                REQUIRE(linRes->hasOperator() == false);

                const auto& gradExpr = func.getGradientExpression();
                REQUIRE(gradExpr.getDataVector() == dc);
                REQUIRE(gradExpr.getOperator() == idOp);
            }

            THEN("a clone behaves as expected") {
                auto qClone = func.clone();

                REQUIRE(qClone.get() != &func);
                REQUIRE(*qClone == func);
            }

            THEN("the evaluate, gradient and Hessian work as expected") {
                RealVector_t dataVec(dd.getNumberOfCoefficients());
                dataVec.setRandom();
                DataContainer x(dd, dataVec);

                real_t trueValue = static_cast<real_t>(0.5) * x.dot(x) - x.dot(dc);
                REQUIRE(func.evaluate(x) == Approx(trueValue));
                DataContainer grad(dd, (dataVec - randomData).eval() );
                REQUIRE(func.getGradient(x) == grad);
                REQUIRE(func.getHessian(x) == leaf(idOp));
            }
        }
    }
}