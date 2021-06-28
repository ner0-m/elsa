/**
 * @file test_WLSProblem.cpp
 *
 * @brief Tests for the WLSProblem class
 *
 * @author Matthias Wieczorek - initial code
 * @author Maximilian Hornung - modularization
 * @author David Frank - rewrite
 * @author Tobias Lasser - rewrite, modernization
 * @author Nikola Dinev - added tests for conversion constructor
 */

#include "doctest/doctest.h"

#include "WLSProblem.h"
#include "Identity.h"
#include "Scaling.h"
#include "L2NormPow2.h"
#include "WeightedL2NormPow2.h"
#include "Quadric.h"
#include "TikhonovProblem.h"
#include "BlockLinearOperator.h"
#include "RandomBlocksDescriptor.h"
#include "VolumeDescriptor.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("WLSProblems: Testing with operator and data", TestType, float, double)
{
    GIVEN("the operator and data")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 13;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(dd, bVec);

        Identity<TestType> idOp(dd);

        WHEN("setting up a ls problem without x0")
        {
            WLSProblem<TestType> prob(idOp, dcB);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<TestType> dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(
                    checkApproxEq(prob.evaluate(), as<TestType>(0.5) * bVec.squaredNorm()));
                REQUIRE_UNARY(
                    checkApproxEq(prob.getGradient(), static_cast<TestType>(-1.0f) * dcB));

                auto hessian = prob.getHessian();
                REQUIRE_UNARY(isApprox(hessian.apply(dcB), dcB));
            }
        }

        WHEN("setting up a ls problem with x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);

            WLSProblem<TestType> prob(idOp, dcB, dcX0);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {

                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcX0));

                REQUIRE_UNARY(checkApproxEq(prob.evaluate(),
                                            as<TestType>(0.5) * (x0Vec - bVec).squaredNorm()));
                REQUIRE_UNARY(isApprox(prob.getGradient(), dcX0 - dcB));

                auto hessian = prob.getHessian();
                REQUIRE_UNARY(isApprox(hessian.apply(dcB), dcB));
            }
        }
    }
}

TEST_CASE_TEMPLATE("WLSProblems: Testing with weights, operator and data", TestType, float, double)
{
    GIVEN("weights, operator and data")
    {
        IndexVector_t numCoeff(3);
        numCoeff << 7, 13, 17;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> bVec(dd.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(dd, bVec);

        Identity<TestType> idOp(dd);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> weightsVec(dd.getNumberOfCoefficients());
        weightsVec.setRandom();
        DataContainer dcWeights(dd, weightsVec);
        Scaling scaleOp(dd, dcWeights);

        WHEN("setting up a wls problem without x0")
        {
            WLSProblem<TestType> prob(scaleOp, idOp, dcB);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<TestType> dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcZero));

                REQUIRE_UNARY(checkApproxEq(
                    prob.evaluate(),
                    as<TestType>(0.5) * bVec.dot((weightsVec.array() * bVec.array()).matrix())));
                DataContainer<TestType> tmpDc = as<TestType>(-1) * dcWeights * dcB;
                REQUIRE_UNARY(isApprox(prob.getGradient(), tmpDc));

                auto hessian = prob.getHessian();
                REQUIRE_UNARY(isApprox(hessian.apply(dcB), dcWeights * dcB));
            }
        }

        WHEN("setting up a wls problem with x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);

            WLSProblem<TestType> prob(scaleOp, idOp, dcB, dcX0);

            THEN("the clone works correctly")
            {
                auto probClone = prob.clone();

                REQUIRE_NE(probClone.get(), &prob);
                REQUIRE_EQ(*probClone, prob);
            }

            THEN("the problem behaves as expected")
            {
                DataContainer<TestType> dcZero(dd);
                dcZero = 0;
                REQUIRE_UNARY(isApprox(prob.getCurrentSolution(), dcX0));

                REQUIRE_UNARY(checkApproxEq(
                    prob.evaluate(),
                    as<TestType>(0.5)
                        * (x0Vec - bVec)
                              .dot((weightsVec.array() * (x0Vec - bVec).array()).matrix())));
                REQUIRE_UNARY(isApprox(prob.getGradient(), dcWeights * (dcX0 - dcB)));

                auto hessian = prob.getHessian();
                REQUIRE_UNARY(isApprox(hessian.apply(dcB), dcWeights * dcB));
            }
        }
    }
}

TEST_CASE_TEMPLATE("WLSProblems: Testing different optimization problems", TestType, float, double)
{
    GIVEN("an optimization problem with only a (w)ls data term")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};

        Scaling<TestType> W{desc, static_cast<TestType>(3.0)};

        LinearResidual<TestType> residual{desc};
        L2NormPow2<TestType> func{residual};
        WeightedL2NormPow2<TestType> weightedFunc{residual, W};
        Problem<TestType> prob{func};
        Problem<TestType> weightedProb{weightedFunc};

        WHEN("converting to a wls problem")
        {
            WLSProblem<TestType> lsProb{prob};
            WLSProblem<TestType> wlsProb{weightedProb};

            THEN("only the type of the problem changes")
            {
                REQUIRE_EQ(lsProb.getDataTerm(), prob.getDataTerm());
                REQUIRE_EQ(lsProb.getRegularizationTerms(), prob.getRegularizationTerms());
                REQUIRE_EQ(wlsProb.getDataTerm(), weightedProb.getDataTerm());
                REQUIRE_EQ(wlsProb.getRegularizationTerms(), weightedProb.getRegularizationTerms());
            }
        }
    }

    GIVEN("an optimization problem with a non-(w)ls data term")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};

        Quadric<TestType> quadric{desc};
        Problem prob{quadric};

        WHEN("converting to a WLSProblem")
        {
            THEN("an exception is thrown") { REQUIRE_THROWS(WLSProblem<TestType>{prob}); }
        }
    }

    GIVEN("an optimization problem with a non-(w)ls regularization term")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};

        Quadric<TestType> quadric{desc};
        RegularizationTerm regTerm{static_cast<TestType>(5), quadric};
        Problem prob{L2NormPow2<TestType>{desc}, regTerm};

        WHEN("converting to a WLSProblem")
        {
            THEN("an exception is thrown") { REQUIRE_THROWS(WLSProblem<TestType>{prob}); }
        }
    }

    GIVEN("an optimization problem with a wls data term that has negative weighting factors")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};

        Scaling<TestType> W1{desc, static_cast<TestType>(-3.0)};

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> anisotropicW =
            Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Constant(343, 1);
        anisotropicW[256] = -3.0;

        Scaling<TestType> W2{desc, DataContainer(desc, anisotropicW)};

        LinearResidual<TestType> residual{desc};
        WeightedL2NormPow2<TestType> weightedFunc1{residual, W1};
        WeightedL2NormPow2<TestType> weightedFunc2{residual, W2};

        WHEN("convering to a WLSProblem and no regularization terms are present")
        {
            Problem prob1{weightedFunc1};
            Problem prob2{weightedFunc2};

            WLSProblem converted1{prob1};
            WLSProblem converted2{prob2};

            THEN("only the type of the problem changes")
            {
                REQUIRE_EQ(prob1.getDataTerm(), converted1.getDataTerm());
                REQUIRE_EQ(prob1.getRegularizationTerms(), converted1.getRegularizationTerms());
                REQUIRE_EQ(prob2.getDataTerm(), converted2.getDataTerm());
                REQUIRE_EQ(prob2.getRegularizationTerms(), converted2.getRegularizationTerms());
            }
        }

        WHEN("convering to a WLSProblem and regularization terms are present")
        {
            RegularizationTerm regTerm{static_cast<TestType>(1.0), L2NormPow2<TestType>{desc}};
            Problem prob1{weightedFunc1, regTerm};
            Problem prob2{weightedFunc2, regTerm};

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS(WLSProblem{prob1});
                REQUIRE_THROWS(WLSProblem{prob2});
            }
        }
    }

    GIVEN("an optimization problem with a (w)ls regularization term that has negative weighting "
          "factors")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};

        Scaling<TestType> W1{desc, static_cast<TestType>(-3.0)};

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> anisotropicW =
            Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Constant(343, 1);
        anisotropicW[256] = -3.0;

        Scaling<TestType> W2{desc, DataContainer(desc, anisotropicW)};

        LinearResidual<TestType> residual{desc};
        WeightedL2NormPow2<TestType> weightedFunc1{residual, W1};
        WeightedL2NormPow2<TestType> weightedFunc2{residual, W2};
        L2NormPow2<TestType> nonWeightedFunc{desc};

        RegularizationTerm negWeights{static_cast<TestType>(1.0), weightedFunc1};
        RegularizationTerm mixedWeights{static_cast<TestType>(1.0), weightedFunc2};
        RegularizationTerm noWeightsNegLambda{static_cast<TestType>(-1.0), nonWeightedFunc};
        WHEN("convering to a WLSProblem")
        {
            Problem prob1{L2NormPow2<TestType>{desc}, negWeights};
            Problem prob2{L2NormPow2<TestType>{desc}, mixedWeights};
            Problem prob3{L2NormPow2<TestType>{desc}, noWeightsNegLambda};

            THEN("an exception is thrown")
            {
                REQUIRE_THROWS(WLSProblem{prob1});
                REQUIRE_THROWS(WLSProblem{prob2});
                REQUIRE_THROWS(WLSProblem{prob3});
            }
        }
    }

    GIVEN("an OptimizationProblem with only (w)ls terms")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec =
            Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343);
        DataContainer<TestType> b{desc, vec};

        Scaling<TestType> A{desc, static_cast<TestType>(2.0)};

        Scaling<TestType> isoW{desc, static_cast<TestType>(3.0)};
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vecW =
            Eigen::abs(Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343).array());
        DataContainer<TestType> dcW{desc, vecW};
        Scaling<TestType> nonIsoW{desc, dcW};

        std::vector<std::unique_ptr<Functional<TestType>>> dataTerms;

        dataTerms.push_back(std::make_unique<L2NormPow2<TestType>>(desc));
        dataTerms.push_back(std::make_unique<L2NormPow2<TestType>>(LinearResidual{b}));
        dataTerms.push_back(std::make_unique<L2NormPow2<TestType>>(LinearResidual{A}));
        dataTerms.push_back(std::make_unique<L2NormPow2<TestType>>(LinearResidual{A, b}));
        dataTerms.push_back(
            std::make_unique<WeightedL2NormPow2<TestType>>(LinearResidual{A, b}, isoW));
        dataTerms.push_back(
            std::make_unique<WeightedL2NormPow2<TestType>>(LinearResidual{A, b}, nonIsoW));

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> regVec =
            Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343);
        DataContainer<TestType> bReg{desc, regVec};

        Scaling<TestType> AReg{desc, static_cast<TestType>(0.25)};

        Scaling<TestType> isoWReg{desc, static_cast<TestType>(1.5)};
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vecWReg =
            Eigen::abs(Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343).array());
        DataContainer<TestType> dcWReg{desc, vecWReg};
        Scaling<TestType> nonIsoWReg{desc, dcWReg};

        std::vector<std::unique_ptr<RegularizationTerm<TestType>>> regTerms;
        auto weight = static_cast<TestType>(0.5);
        regTerms.push_back(
            std::make_unique<RegularizationTerm<TestType>>(weight, L2NormPow2<TestType>{desc}));
        regTerms.push_back(std::make_unique<RegularizationTerm<TestType>>(
            weight, L2NormPow2<TestType>{LinearResidual{bReg}}));
        regTerms.push_back(std::make_unique<RegularizationTerm<TestType>>(
            weight, L2NormPow2<TestType>{LinearResidual{AReg}}));
        regTerms.push_back(std::make_unique<RegularizationTerm<TestType>>(
            weight, L2NormPow2<TestType>{LinearResidual{AReg, bReg}}));
        regTerms.push_back(std::make_unique<RegularizationTerm<TestType>>(
            weight, WeightedL2NormPow2{LinearResidual{AReg, bReg}, isoWReg}));
        regTerms.push_back(std::make_unique<RegularizationTerm<TestType>>(
            weight, WeightedL2NormPow2{LinearResidual{AReg, bReg}, nonIsoWReg}));

        std::array descriptions = {"has no operator and no vector",
                                   "has no operator, but has a vector",
                                   "has an operator, but no vector",
                                   "has an operator and a vector",
                                   "has an operator and a vector, and is weighted (isotropic)",
                                   "has an operator and a vector, and is weighted (nonisotropic)"};

        for (std::size_t i = 0; i < descriptions.size(); i++) {
            for (std::size_t j = 0; j < descriptions.size(); j++) {
                WHEN("Trying different settings")
                {
                    INFO("The data term ", descriptions[i]);
                    INFO("The regularization term ", descriptions[j]);
                    Eigen::Matrix<TestType, Eigen::Dynamic, 1> xVec =
                        Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343);
                    DataContainer<TestType> x{desc, xVec};
                    Problem prob{*dataTerms[i], *regTerms[j], x};

                    THEN("the problem can be converted and all operations yield the same result as "
                         "for the initial problem")
                    {
                        WLSProblem<TestType> converted{prob};
                        REQUIRE_UNARY(checkApproxEq(prob.evaluate(), converted.evaluate()));

                        auto gradDiff = prob.getGradient();
                        gradDiff -= converted.getGradient();
                        REQUIRE_UNARY(checkApproxEq(gradDiff.squaredL2Norm(), 0));
                    }
                }
            }
        }
    }

    GIVEN("a TikhonovProblem with L2 regularization")
    {
        VolumeDescriptor desc{IndexVector_t::Constant(1, 343)};
        Eigen::Matrix<TestType, Eigen::Dynamic, 1> vec =
            Eigen::Matrix<TestType, Eigen::Dynamic, 1>::Random(343);
        DataContainer<TestType> b{desc, vec};

        Scaling<TestType> A{desc, static_cast<TestType>(2.0)};
        WLSProblem<TestType> prob{A, b};

        TestType regWeight = 4.0;
        TikhonovProblem<TestType> l2Reg{prob,
                                        RegularizationTerm{regWeight, L2NormPow2<TestType>{desc}}};

        THEN("the problem can be converted into a block form WLSProblem")
        {
            WLSProblem<TestType> conv{l2Reg};

            Scaling<TestType> lambdaScaling(desc, std::sqrt(regWeight));
            std::vector<std::unique_ptr<LinearOperator<TestType>>> opList(0);
            opList.push_back(A.clone());
            opList.push_back(lambdaScaling.clone());
            BlockLinearOperator<TestType> blockOp{opList,
                                                  BlockLinearOperator<TestType>::BlockType::ROW};

            std::vector<std::unique_ptr<DataDescriptor>> descList(0);
            descList.push_back(desc.clone());
            descList.push_back(desc.clone());
            RandomBlocksDescriptor vecDesc{descList};
            DataContainer<TestType> blockVec{vecDesc};
            blockVec.getBlock(0) = b;
            blockVec.getBlock(1) = 0;

            L2NormPow2<TestType> blockWls{LinearResidual<TestType>{blockOp, blockVec}};
            REQUIRE_EQ(conv.getDataTerm(), blockWls);
            REQUIRE_UNARY(conv.getRegularizationTerms().empty());
        }
    }
}

TEST_SUITE_END();
