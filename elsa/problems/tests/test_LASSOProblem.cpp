/**
 * @file test_LASSOProblem.cpp
 *
 * @brief Tests for the LASSOProblem class
 *
 * @author Andi Braimllari
 */

#include "doctest/doctest.h"

#include "Error.h"
#include "L2NormPow2.h"
#include "LASSOProblem.h"
#include "VolumeDescriptor.h"
#include "Identity.h"
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("Scenario: Testing LASSOProblem", data_t, float, double)
{
    GIVEN("some data term and a regularization term")
    {
        VolumeDescriptor dd({17, 53});

        Vector_t<data_t> scaling(dd.getNumberOfCoefficients());
        scaling.setRandom();
        DataContainer<data_t> dcScaling(dd, scaling);
        Scaling scaleOp(dd, dcScaling);

        Vector_t<data_t> dataVec(dd.getNumberOfCoefficients());
        dataVec.setRandom();
        DataContainer<data_t> dcData(dd, dataVec);

        WLSProblem<data_t> wlsProblem(scaleOp, dcData);

        auto invalidWeight = static_cast<data_t>(-2.0);
        auto weight = static_cast<data_t>(2.0);

        // l1 norm regularization term
        L1Norm<data_t> regFunc(dd);

        WHEN("setting up a LASSOProblem with a negative regularization weight")
        {
            RegularizationTerm<data_t> invalidRegTerm(invalidWeight, regFunc);
            THEN("an invalid_argument exception is thrown")
            {
                REQUIRE_THROWS_AS(LASSOProblem<data_t>(wlsProblem, invalidRegTerm),
                                  InvalidArgumentError);
            }
        }

        // l2 norm regularization term
        L2NormPow2<data_t> invalidRegFunc(dd);

        WHEN("setting up a LASSOProblem with a L2NormPow2 regularization term")
        {
            RegularizationTerm<data_t> invalidRegTerm(weight, invalidRegFunc);
            THEN("an invalid_argument exception is thrown")
            {
                REQUIRE_THROWS_AS(LASSOProblem<data_t>(wlsProblem, invalidRegTerm),
                                  InvalidArgumentError);
            }
        }

        RegularizationTerm<data_t> regTerm(weight, regFunc);

        WHEN("setting up a LASSOProblem without an x0")
        {
            LASSOProblem<data_t> lassoProb(wlsProblem, regTerm);

            THEN("cloned LASSOProblem equals original LASSOProblem")
            {
                auto lassoProbClone = lassoProb.clone();

                REQUIRE_NE(lassoProbClone.get(), &lassoProb);
                REQUIRE_EQ(*lassoProbClone, lassoProb);
            }
        }

        WHEN("setting up a LASSOProblem with an x0")
        {
            Eigen::Matrix<data_t, Eigen::Dynamic, 1> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<data_t> dcX0(dd, x0Vec);

            wlsProblem.getCurrentSolution() = dcX0;
            LASSOProblem<data_t> lassoProb(wlsProblem, regTerm);

            THEN("cloned LASSOProblem equals original LASSOProblem")
            {
                auto lassoProbClone = lassoProb.clone();

                REQUIRE_NE(lassoProbClone.get(), &lassoProb);
                REQUIRE_EQ(*lassoProbClone, lassoProb);
            }
        }

        Identity<data_t> idOp(dd);
        WLSProblem<data_t> wlsProblemForLC(idOp, dcData);

        WHEN("setting up the Lipschitz Constant of a LASSOProblem without an x0")
        {
            LASSOProblem<data_t> lassoProb(wlsProblemForLC, regTerm);

            auto lipschitzConstant = lassoProb.getLipschitzConstant();

            THEN("the Lipschitz Constant of a LASSOProblem with an Identity Operator as the "
                 "Linear Operator A is 1")
            {
                REQUIRE_UNARY(checkApproxEq(lipschitzConstant, as<data_t>(1.0)));
            }
        }

        WHEN("setting up the Lipschitz Constant of a LASSOProblem with an x0")
        {
            Vector_t<data_t> x0Vec(dd.getNumberOfCoefficients());
            x0Vec.setRandom();
            DataContainer<data_t> dcX0(dd, x0Vec);
            wlsProblemForLC.getCurrentSolution() = dcX0;

            LASSOProblem<data_t> lassoProb(wlsProblemForLC, regTerm);

            auto lipschitzConstant = lassoProb.getLipschitzConstant();

            THEN("the Lipschitz Constant of a LASSOProblem with an Identity Operator as the "
                 "Linear Operator A is 1")
            {
                REQUIRE_EQ(lipschitzConstant, as<data_t>(1.0));
            }
        }
    }
}

TEST_SUITE_END();
