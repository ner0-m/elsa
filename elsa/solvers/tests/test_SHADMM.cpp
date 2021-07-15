/**
 * @file test_SHADMM.cpp
 *
 * @brief Tests for the SHADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "SHADMM.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"
#include "VolumeDescriptor.h"
#include "BlockLinearOperator.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("SHADMM: Solving problems", data_t, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 7, 7;
        VolumeDescriptor volDescr(numCoeff);

        Vector_t<data_t> bVec(volDescr.getNumberOfCoefficients());
        bVec.setRandom();
        DataContainer<data_t> dcB(volDescr, bVec);

        Identity<data_t> idOp(volDescr);

        WLSProblem<data_t> wlsProb(idOp, dcB);

        Identity<data_t> idOp1(volDescr);
        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfA(0);
        opsOfA.push_back(std::move(idOp1.clone())); /// SH
        opsOfA.push_back(std::move(idOp1.clone()));
        BlockLinearOperator<data_t> A{opsOfA, BlockLinearOperator<data_t>::BlockType::COL};

        std::vector<std::unique_ptr<LinearOperator<data_t>>> opsOfB(0);
        opsOfB.push_back(std::move(idOp1.clone()));
        opsOfB.push_back(std::move(idOp1.clone()));
        BlockLinearOperator<data_t> B{opsOfB, BlockLinearOperator<data_t>::BlockType::COL};

        DataContainer<data_t> dCC(volDescr);
        dCC = 0;

        Constraint<data_t> constraint(A, B, dCC);

        WHEN("setting up SHADMM to solve a problem")
        {
            DataContainer<data_t> ones(volDescr);
            ones = 1;
            WeightedL1Norm weightedL1Norm(ones);
            RegularizationTerm<data_t> wL1NormRegTerm(1, weightedL1Norm);

            Indicator<data_t> indicator(volDescr);
            RegularizationTerm<data_t> indicRegTerm(1, indicator);

            SplittingProblem<data_t> splittingProblem(
                wlsProb.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

            SHADMM<CG, data_t> shadmm(splittingProblem);

            THEN("the solution doesn't throw, is not nan and is approximate to the bvector")
            {
                REQUIRE_NOTHROW(shadmm.solve(10));
                printf("%f\n", shadmm.solve(20).squaredL2Norm());
                REQUIRE_UNARY(!std::isnan(shadmm.solve(20).squaredL2Norm()));
                REQUIRE_UNARY(isApprox(shadmm.solve(20), dcB));
            }
        }
    }
}

TEST_SUITE_END();
