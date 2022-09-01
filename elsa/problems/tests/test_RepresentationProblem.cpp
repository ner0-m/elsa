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
#include "testHelpers.h"

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("problems");

TEST_CASE_TEMPLATE("RepresentationProblem: Setup with a dictionary and a signal vector", data_t,
                   float, double)
{
    GIVEN("some dictionary operator and a signal vector")
    {
        const index_t nAtoms = 10;
        VolumeDescriptor signalDescriptor({5});

        Dictionary<data_t> dictOp(signalDescriptor, nAtoms);

        Vector_t<data_t> signalVec(signalDescriptor.getNumberOfCoefficients());
        signalVec.setRandom();
        DataContainer<data_t> dcSignal(signalDescriptor, signalVec);

        RepresentationProblem<data_t> reprProblem(dictOp, dcSignal);

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
            Vector_t<data_t> reprVec(reprDescriptor.getNumberOfCoefficients());
            reprVec.setRandom();
            DataContainer<data_t> dcRepresentation(reprDescriptor, reprVec);

            auto evaluation = reprProblem.evaluate(dcRepresentation);

            THEN("the evaluation is as expected")
            {
                DataContainer<data_t> residual = (dcSignal - dictOp.apply(dcRepresentation));
                data_t expected = as<data_t>(0.5) * residual.squaredL2Norm();

                REQUIRE_UNARY(checkApproxEq(evaluation, expected));
            }
        }

        WHEN("getting the dictionary back")
        {
            const auto& reprProblemDict = reprProblem.getDictionary();

            THEN("it equals the original dictionary")
            {
                REQUIRE_EQ(reprProblemDict, dictOp);
            }
        }

        WHEN("getting the signal back")
        {
            auto reprProblemSignal = reprProblem.getSignal();

            THEN("it equals the original signal")
            {
                REQUIRE_EQ(reprProblemSignal, dcSignal);
            }
        }
    }
}

TEST_SUITE_END();
