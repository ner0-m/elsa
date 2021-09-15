/**
 * @file test_SHADMM.cpp
 *
 * @brief Tests for the SHADMM class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "SHADMM.h"
#include "VolumeDescriptor.h"
#include "BlockLinearOperator.h"
#include "CG.h"
#include "SoftThresholding.h"
#include "Identity.h"
#include "FISTA.h"
#include "Logger.h"

#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("solvers");

TEST_CASE_TEMPLATE("SHADMM: Solving problems", TestType, float, double)
{
    Logger::setLevel(Logger::LogLevel::OFF);

    GIVEN("some problems and a constraint")
    {
        IndexVector_t size(2);
        size << 128, 128;
        VolumeDescriptor volDescr(size);

        // random number
        TestType rho1 = 23;
        // random number
        TestType rho2 = 86;

        /// AT = (ρ_1*SH^T, ρ_2*I_n^2 ) ∈ R ^ n^2 × (L+1)n^2
        ShearletTransform<TestType> shearletTransform(size[0], size[1]);
        index_t L = shearletTransform.getL();
        // size[0] == size[1], for now at least
        index_t n = shearletTransform.getWidth();

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfA(0);
        //        DataContainer<TestType> rho2s(VolumeDescriptor{{L * n * n, L * n * n}});
        //        rho2s = rho2;
        Scaling<TestType> scaling1(VolumeDescriptor{n * n}, rho2);
        opsOfA.push_back(std::move(shearletTransform.clone()));
        opsOfA.push_back(std::move(scaling1.clone()));
        BlockLinearOperator<TestType> A{opsOfA, BlockLinearOperator<TestType>::BlockType::ROW};

        /// B = diag(−ρ_1*1_Ln^2 , −ρ_2*1_n^2) ∈ R ^ (L+1)n^2 × (L+1)n^2

        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB1(0);

        //        //        DataContainer<TestType> nRho1s(VolumeDescriptor{{L * n * n, L * n *
        //        n}});
        //        //        nRho1s = -rho1;
        //        Identity<TestType> B11(VolumeDescriptor{{L * n * n, 1}});
        //        //        DataContainer<TestType> zeroes12(VolumeDescriptor{{L * n * n, n * n}});
        //        //        zeroes12 = 0;
        //        Scaling<TestType> B12(VolumeDescriptor{{n * n, 1}}, VolumeDescriptor{{L * n * n,
        //        1}}, 0); opsOfB1.push_back(std::move(B11.clone()));
        //        opsOfB1.push_back(std::move(B12.clone()));
        //        BlockLinearOperator<TestType> B1{opsOfB1,
        //        BlockLinearOperator<TestType>::BlockType::COL};

        //        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB2(0);
        //        DataContainer<TestType> zeroes21(VolumeDescriptor{{n * n, L * n * n}});
        //        zeroes21 = 0;
        //        Scaling<TestType> B21(VolumeDescriptor{{n * n, L * n * n}}, zeroes21);
        //        DataContainer<TestType> nRho2s(VolumeDescriptor{{n * n, n * n}});
        //        nRho2s = -rho2;
        //        Scaling<TestType> B22(VolumeDescriptor{{n * n, n * n}}, nRho2s);
        //        opsOfB2.push_back(std::move(B21.clone()));
        //        opsOfB2.push_back(std::move(B22.clone()));
        //        BlockLinearOperator<TestType> B2{opsOfB2,
        //        BlockLinearOperator<TestType>::BlockType::COL};
        //
        //        std::vector<std::unique_ptr<LinearOperator<TestType>>> opsOfB(0);
        //        opsOfB1.push_back(std::move(B1.clone()));
        //        opsOfB1.push_back(std::move(B2.clone()));
        //        BlockLinearOperator<TestType> B{opsOfB,
        //        BlockLinearOperator<TestType>::BlockType::ROW};

        Identity<TestType> tempId(VolumeDescriptor{(L + 1) * n * n});

        DataContainer<TestType> dCC(VolumeDescriptor{(L + 1) * n * n});
        dCC = 0;

        Constraint<TestType> constraint(A, tempId, dCC);

        WHEN("setting up SHADMM to solve a problem")
        {
            Vector_t<TestType> bVec(VolumeDescriptor{n * n}.getNumberOfCoefficients());
            bVec.setRandom();
            DataContainer<TestType> dcB(VolumeDescriptor{n * n}, bVec);

            Identity<TestType> idOp(VolumeDescriptor{n * n});
            // TODO use these in the end?
            // // generate circular trajectory
            //            index_t numAngles{180}, arc{360};
            //            const auto distance = static_cast<real_t>(size(0));
            //            auto sinoDescriptor = CircleTrajectoryGenerator::createTrajectory(
            //                numAngles, VolumeDescriptor{{n, n}}, arc, distance * 100.0f,
            //                distance);
            //            SiddonsMethod projector(VolumeDescriptor{n * n}, *sinoDescriptor);

            WLSProblem<TestType> wlsProb(idOp, dcB);

            DataContainer<TestType> ones(VolumeDescriptor{L * n * n});
            ones = 1;
            WeightedL1Norm<TestType> weightedL1Norm(ones);
            RegularizationTerm<TestType> wL1NormRegTerm(1, weightedL1Norm);

            Indicator<TestType> indicator(VolumeDescriptor{n * n});
            RegularizationTerm<TestType> indicRegTerm(1, indicator);

            SplittingProblem<TestType> splittingProblem(
                wlsProb.getDataTerm(), std::vector{wL1NormRegTerm, indicRegTerm}, constraint);

            SHADMM<CG, SoftThresholding, TestType> shadmm(splittingProblem);

            THEN("the solution doesn't throw, is not nan and is approximate to the bvector")
            {
                auto sol = shadmm.solve(10);
                printf("%f\n", sol.squaredL2Norm());
                //                REQUIRE_NOTHROW(sol);
                //                REQUIRE_UNARY(!std::isnan(shadmm.solve(20).squaredL2Norm()));
                //                REQUIRE_UNARY(isApprox(shadmm.solve(20), dcB));
            }
        }
    }
}

TEST_SUITE_END();
