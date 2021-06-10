/**
 * @file test_RepresentationProblem.cpp
 *
 * @brief Tests for the RepresentationProblem class
 *
 * @author Jonas Buerger
 */

#include "doctest/doctest.h"

#include "RepresentationProblem.h"
#include "VolumeDescriptor.h"
#include "Dictionary.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("Scenario: Testing RepresentationProblem", TestType, float, double)
{
    GIVEN("some dictionary operator and a signal vector")
    {
        const index_t nAtoms = 10;
        VolumeDescriptor signalDescriptor({5});

        Dictionary<TestType> dictOp(signalDescriptor, nAtoms);

        Eigen::Matrix<TestType, Eigen::Dynamic, 1> signalVec(
            signalDescriptor.getNumberOfCoefficients());
        signalVec.setRandom();
        DataContainer<TestType> dcSignal(signalDescriptor, signalVec);

        RepresentationProblem<TestType> reprProblem(dictOp, dcSignal);

        WHEN("cloning a RepresentationProblem")
        {
            auto reprProblemClone = reprProblem.clone();
            THEN("cloned RepresentationProblem equals original RepresentationProblem")
            {
                REQUIRE_NE(reprProblemClone.get(), &reprProblem);
                REQUIRE_EQ(*reprProblemClone, reprProblem);
            }
        }

        WHEN("evaluating the problem for a random representation")
        {
            VolumeDescriptor reprDescriptor({nAtoms});
            Eigen::Matrix<TestType, Eigen::Dynamic, 1> reprVec(
                reprDescriptor.getNumberOfCoefficients());
            reprVec.setRandom();
            DataContainer<TestType> dcRepresentation(reprDescriptor, reprVec);

            reprProblem.getCurrentSolution() = dcRepresentation;
            auto evaluation = reprProblem.evaluate();

            THEN("the evaluation is as expected")
            {
                DataContainer<TestType> residual = (dcSignal - dictOp.apply(dcRepresentation));
                auto expected = 0.5 * residual.squaredL2Norm();
                REQUIRE_EQ(evaluation, Approx(expected));
            }
        }

        WHEN("getting the dictionary back")
        {
            const auto& reprProblemDict = reprProblem.getDictionary();

            THEN("it equals the original dictionary") { REQUIRE_EQ(reprProblemDict, dictOp); }
        }

        WHEN("getting the signal back")
        {
            auto reprProblemSignal = reprProblem.getSignal();

            THEN("it equals the original signal") { REQUIRE_EQ(reprProblemSignal, dcSignal); }
        }
    }
}

TEST_SUITE_END();
