/**
 * @file test_PhantomNet.cpp
 *
 * @brief Tests for the PhantomNet class
 *
 * @author Andi Braimllari
 */

#include "PhantomNet.h"
#include "VolumeDescriptor.h"
#include "Error.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

TEST_CASE_TEMPLATE("PhantomNet: Pre-training testing", TestType, float, double)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor mnistShape({28, 28});
        index_t batchSize = 32;

        WHEN("instantiating the PhantomNet model")
        {
            ml::PhantomNet<TestType, elsa::ml::MlBackend::Cudnn> aeModel(mnistShape, batchSize);

            THEN("the shape of the input matches the shape of the output")
            {
                DataContainer<TestType> randomValues(mnistShape);
                DataContainer<TestType> prediction = aeModel.predict(randomValues);
                REQUIRE_EQ(prediction.getDataDescriptor(), mnistShape);
            }
        }
    }
}

TEST_SUITE_END();
