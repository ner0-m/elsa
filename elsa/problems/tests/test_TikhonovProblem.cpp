/**
 * \file test_TikhonovProblem.cpp
 *
 * \brief Tests for the TikhonovProblem<TestType> class
 *
 * \author Nikola Dinev
 */

#include <catch2/catch.hpp>
#include "Problem.h"
#include "Identity.h"
#include "Scaling.h"
#include "LinearResidual.h"
#include "L2NormPow2.h"
#include "L1Norm.h"
#include "TikhonovProblem.h"

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing TikhonovProblem with one regularization term", "", float,
                   double)
{
    GIVEN("some data term and some regularization term")
    {
        // least squares data term
        IndexVector_t numCoeff(2);
        numCoeff << 23, 47;
        DataDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<TestType> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<TestType> dcData(dd, dataVec);

        WLSProblem<TestType> wls(scaleOp, dcData);

        // l2 norm regularization term
        L2NormPow2<TestType> regFunc(dd);
        TestType weight = 2.0;
        RegularizationTerm<TestType> regTerm(weight, regFunc);

        WHEN("setting up a TikhonovProblem without regularization terms")
        {
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(
                    TikhonovProblem<TestType>(wls, std::vector<RegularizationTerm<TestType>>{}),
                    std::invalid_argument);
            }
        }

        WHEN("setting up a TikhonovProblem with a non (Weighted)L2NormPow2 regularization term")
        {
            L1Norm<TestType> invalidRegFunc(dd);
            RegularizationTerm<TestType> invalidRegTerm(1.0, invalidRegFunc);
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(TikhonovProblem<TestType>(wls, invalidRegTerm),
                                  std::invalid_argument);
            }
        }

        WHEN("setting up the TikhonovProblem without x0")
        {
            TikhonovProblem<TestType> prob(wls, regTerm);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                DataContainer<TestType> dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * dataVec.squaredNorm()));
                REQUIRE(prob.getGradient() == static_cast<TestType>(-1.0) * dcScaling * dcData);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE(prob != optProb);
                REQUIRE(optProb != prob);
            }
        }

        WHEN("setting up the TikhonovProblem with x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);

            wls.getCurrentSolution() = dcX0;
            TikhonovProblem<TestType> prob(wls, regTerm);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                auto valueData =
                    0.5
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE(prob.evaluate() == Approx(valueData + weight * 0.5 * x0Vec.squaredNorm()));
                REQUIRE(prob.getGradient()
                        == dcScaling * (dcScaling * dcX0 - dcData) + weight * dcX0);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE(prob != optProb);
                REQUIRE(optProb != prob);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Scenario: Testing TikhonovProblem with several regularization terms", "", float,
                   double)
{
    GIVEN("some data term and several regularization terms")
    {
        // least squares data term
        IndexVector_t numCoeff(3);
        numCoeff << 17, 33, 52;
        DataDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<TestType> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<TestType> dcData(dd, dataVec);

        WLSProblem<TestType> wls(scaleOp, dcData);

        // l2 norm regularization term
        L2NormPow2<TestType> regFunc(dd);
        TestType weight1 = 2.0;
        RegularizationTerm<TestType> regTerm1(weight1, regFunc);

        TestType weight2 = 3.0;
        RegularizationTerm<TestType> regTerm2(weight2, regFunc);

        std::vector<RegularizationTerm<TestType>> vecReg{regTerm1, regTerm2};

        WHEN("setting up a TikhonovProblem with a non (Weighted)L2NormPow2 regularization term")
        {
            L1Norm<TestType> invalidRegFunc(dd);
            RegularizationTerm<TestType> invalidRegTerm(1.0, invalidRegFunc);
            std::vector<RegularizationTerm<TestType>> invalidVecReg1{regTerm1, invalidRegTerm};
            std::vector<RegularizationTerm<TestType>> invalidVecReg2{invalidRegTerm, regTerm2};
            THEN("an exception is thrown")
            {
                REQUIRE_THROWS_AS(TikhonovProblem<TestType>(wls, invalidVecReg1),
                                  std::invalid_argument);
                REQUIRE_THROWS_AS(TikhonovProblem<TestType>(wls, invalidVecReg2),
                                  std::invalid_argument);
            }
        }

        WHEN("setting up the TikhonovProblem without x0")
        {
            TikhonovProblem<TestType> prob(wls, vecReg);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                DataContainer<TestType> dcZero(dd);
                dcZero = 0;
                REQUIRE(prob.getCurrentSolution() == dcZero);

                REQUIRE(prob.evaluate() == Approx(0.5 * dataVec.squaredNorm()));
                REQUIRE(prob.getGradient() == static_cast<TestType>(-1.0) * dcScaling * dcData);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight1 * dataVec[i]
                                      + weight2 * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE(prob != optProb);
                REQUIRE(optProb != prob);
            }
        }

        WHEN("setting up the TikhonovProblem with x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);

            wls.getCurrentSolution() = dcX0;
            TikhonovProblem<TestType> prob(wls, vecReg);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE(probClone.get() != &prob);
                REQUIRE(*probClone == prob);
            }

            THEN("the TikhonovProblem behaves as expected")
            {
                REQUIRE(prob.getCurrentSolution() == dcX0);

                auto valueData =
                    0.5
                    * (scaling.array() * x0Vec.array() - dataVec.array()).matrix().squaredNorm();
                REQUIRE(prob.evaluate()
                        == Approx(valueData + weight1 * 0.5 * x0Vec.squaredNorm()
                                  + weight2 * 0.5 * x0Vec.squaredNorm()));
                REQUIRE(prob.getGradient()
                        == dcScaling * (dcScaling * dcX0 - dcData) + weight1 * dcX0
                               + weight2 * dcX0);

                auto hessian = prob.getHessian();
                auto result = hessian.apply(dcData);
                for (index_t i = 0; i < result.getSize(); ++i)
                    REQUIRE(result[i]
                            == Approx(scaling[i] * scaling[i] * dataVec[i] + weight1 * dataVec[i]
                                      + weight2 * dataVec[i]));
            }

            THEN("the TikhonovProblem is different from a Problem with the same terms")
            {
                Problem optProb(prob.getDataTerm(), prob.getRegularizationTerms());
                REQUIRE(prob != optProb);
                REQUIRE(optProb != prob);
            }
        }
    }
}