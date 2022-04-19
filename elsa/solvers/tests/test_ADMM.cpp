/**
 * @file test_ADMM.cpp
 *
 * @brief Tests for the ADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "ADMM.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "HardThresholding.h"
#include "FISTA.h"
#include "Identity.h"
#include "PartitionDescriptor.h"

#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("ADMM: Solving problems", TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 21, 11;
        VolumeDescriptor volDescr(numCoeff);

        Vector_t<TestType> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(volDescr, bVec);

        Identity<TestType> idOp(volDescr);
        Scaling<TestType> negativeIdOp(volDescr, -1);
        DataContainer<TestType> dCC(volDescr);
        dCC = 0;

        WLSProblem<TestType> wlsProb(idOp, dcB);

        Constraint<TestType> constraint(idOp, negativeIdOp, dCC);

        WHEN("setting up ADMM and FISTA to solve a LASSOProblem")
        {
            L1Norm<TestType> regFunc(volDescr);
            RegularizationTerm<TestType> regTerm(0.000001f, regFunc);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, SoftThresholding, TestType> admm(splittingProblem, 5, 1e-5f, 1e-5f, true,
                                                      false);

            LASSOProblem<TestType> lassoProb(wlsProb, regTerm);
            FISTA<TestType> fista(lassoProb);

            THEN("the solutions match")
            {
                REQUIRE_UNARY(isApprox(admm.solve(100), fista.solve(100)));
            }
        }

        WHEN("setting up ADMM to solve a WLSProblem + L0PseudoNorm")
        {
            L0PseudoNorm<TestType> regFunc(volDescr);
            RegularizationTerm<TestType> regTerm(0.000001f, regFunc);

            SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), regTerm, constraint);

            ADMM<CG, HardThresholding, TestType> admm(splittingProblem, 5, 1e-5f, 1e-5f, true,
                                                      false);

            THEN("the solution doesn't throw, is not nan and is approximate to the b vector")
            {
                REQUIRE_NOTHROW(admm.solve(10));
                REQUIRE_UNARY(!std::isnan(admm.solve(20).squaredL2Norm()));
                REQUIRE_UNARY(isApprox(admm.solve(20), dcB));
            }
        }
    }
}

TEST_CASE_TEMPLATE("ADMM: Solving problems with solutions restricted to only positive values",
                   TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        index_t n = 32;
        IndexVector_t size(2);
        size << n, n;
        VolumeDescriptor volDescr(size);

        Vector_t<TestType> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<TestType> dcB(volDescr, bVec);

        Identity<TestType> idOp(volDescr);

        WLSProblem<TestType> wlsProb(idOp, dcB);

        TestType rho1 = 1.0 / 2;
        TestType rho2 = 1;

        ShearletTransform<TestType, TestType> shearletTransform(size);
        index_t numOfLayers = shearletTransform.getNumOfLayers();

        VolumeDescriptor numOfLayersPlusOneDescriptor{{n, n, numOfLayers + 1}};

        IndexVector_t slicesInBlock(2);
        slicesInBlock << numOfLayers, 1;
        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfA(0);
        Scaling<TestType> scaling(volDescr, rho2);
        opsOfA.push_back((rho1 * shearletTransform).clone());
        opsOfA.push_back(scaling.clone());
        BlockLinearOperator<TestType> A(
            volDescr, PartitionDescriptor{numOfLayersPlusOneDescriptor, slicesInBlock}, opsOfA,
            BlockLinearOperator<TestType>::BlockType::ROW);

        DataContainer<TestType> factorsOfB(numOfLayersPlusOneDescriptor);
        //        factorsOfB.slice(0, numOfLayers) = -1 * rho1;
        //        factorsOfB.slice(numOfLayers) = -1 * rho2;
        for (int ind = 0; ind < factorsOfB.getSize(); ++ind) {
            if (ind < (n * n * numOfLayers)) {
                factorsOfB[ind] = -1 * rho1;
            } else {
                factorsOfB[ind] = -1 * rho2;
            }
        }
        Scaling<TestType> B(numOfLayersPlusOneDescriptor, factorsOfB);

        DataContainer<TestType> c(numOfLayersPlusOneDescriptor);
        c = 0;

        Constraint<TestType> constraint(A, B, c);

        DataContainer<TestType> wL1NWeights(VolumeDescriptor{{n, n, numOfLayers}});
        wL1NWeights = 0.001f;
        //        for (int j = 1; j < layers; ++j) {
        //            wL1NWeights.slice(j - 1, j) = j / 5;
        //        }
        WeightedL1Norm<TestType> weightedL1Norm(LinearResidual<TestType>{shearletTransform},
                                                wL1NWeights);
        RegularizationTerm<TestType> wL1NormRegTerm(1, weightedL1Norm);

        SplittingProblem<TestType> splittingProblem(wlsProb.getDataTerm(), wL1NormRegTerm,
                                                    constraint);

        WHEN("setting up ADMM to solve a problem")
        {
            ADMM<CG, SoftThresholding, TestType> admm(splittingProblem, true);

            THEN("the solution doesn't throw, is not nan, and is approximate to the bvector")
            {
                REQUIRE_NOTHROW(admm.solve(1));
                REQUIRE_UNARY(!std::isnan(admm.solve(2).squaredL2Norm()));
            }
        }
    }
}

TEST_SUITE_END();
