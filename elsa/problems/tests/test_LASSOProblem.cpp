/**
 * @file test_LASSOProblem.cpp
 *
 * @brief Tests for the LASSOProblem class
 *
 * @author Andi Braimllari
 */

#include "L2NormPow2.h"
#include "LASSOProblem.h"
#include "VolumeDescriptor.h"
#include "Identity.h"

#include <catch2/catch.hpp>

using namespace elsa;

TEMPLATE_TEST_CASE("Scenario: Testing LASSOProblem", "", float, double)
{
    GIVEN("some data term and a regularization term")
    {
        IndexVector_t numCoeff(2);
        numCoeff << 17, 53;
        VolumeDescriptor dd(numCoeff);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<TestType> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<TestType> dcData(dd, dataVec);

        WLSProblem<TestType> wlsProblem(scaleOp, dcData);

        auto invalidWeight = static_cast<TestType>(-2.0);
        auto weight = static_cast<TestType>(2.0);

        // l1 norm regularization term
        L1Norm<TestType> regFunc(dd);

        WHEN("setting up a LASSOProblem with a negative regularization weight")
        {
            RegularizationTerm<TestType> invalidRegTerm(invalidWeight, regFunc);
            THEN("an invalid_argument exception is thrown")
            {
                REQUIRE_THROWS_AS(LASSOProblem<TestType>(wlsProblem, invalidRegTerm),
                                  std::invalid_argument);
            }
        }

        // l2 norm regularization term
        L2NormPow2<TestType> invalidRegFunc(dd);

        WHEN("setting up a LASSOProblem with a L2NormPow2 regularization term")
        {
            RegularizationTerm<TestType> invalidRegTerm(weight, invalidRegFunc);
            THEN("an invalid_argument exception is thrown")
            {
                REQUIRE_THROWS_AS(LASSOProblem<TestType>(wlsProblem, invalidRegTerm),
                                  std::invalid_argument);
            }
        }

        RegularizationTerm<TestType> regTerm(weight, regFunc);

        WHEN("setting up a LASSOProblem without an x0")
        {
            LASSOProblem<TestType> lassoProb(wlsProblem, regTerm);

            THEN("cloned LASSOProblem equals original LASSOProblem")
            {
                auto lassoProbClone = lassoProb.clone();

                REQUIRE(lassoProbClone.get() != &lassoProb);
                REQUIRE(*lassoProbClone == lassoProb);
            }
        }

        WHEN("setting up a LASSOProblem with an x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);

            wlsProblem.getCurrentSolution() = dcX0;
            LASSOProblem<TestType> lassoProb(wlsProblem, regTerm);

            THEN("cloned LASSOProblem equals original LASSOProblem")
            {
                auto lassoProbClone = lassoProb.clone();

                REQUIRE(lassoProbClone.get() != &lassoProb);
                REQUIRE(*lassoProbClone == lassoProb);
            }
        }

        Identity<TestType> idOp(dd);
        WLSProblem<TestType> wlsProblemForLC(idOp, dcData);

        WHEN("setting up the Lipschitz Constant of a LASSOProblem without an x0")
        {
            LASSOProblem<TestType> lassoProb(wlsProblemForLC, regTerm);

            TestType lipschitzConstant = lassoProb.getLipschitzConstant();

            THEN("the Lipschitz Constant of a LASSOProblem with an Identity Operator as the "
                 "Linear Operator A is 1")
            {
                REQUIRE(lipschitzConstant == static_cast<TestType>(1.0));
            }
        }

        WHEN("setting up the Lipschitz Constant of a LASSOProblem with an x0")
        {
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<TestType> dcX0(dd, x0Vec);
            wlsProblemForLC.getCurrentSolution() = dcX0;

            LASSOProblem<TestType> lassoProb(wlsProblemForLC, regTerm);

            TestType lipschitzConstant = lassoProb.getLipschitzConstant();

            THEN("the Lipschitz Constant of a LASSOProblem with an Identity Operator as the "
                 "Linear Operator A is 1")
            {
                REQUIRE(lipschitzConstant == static_cast<TestType>(1.0));
            }
        }
    }
}
