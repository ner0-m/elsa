/**
 * @file test_AutoEncoder.cpp
 *
 * @brief Tests for the AutoEncoder class
 *
 * @author Andi Braimllari
 */

#include "AutoEncoder.h"
#include "VolumeDescriptor.h"
#include "Error.h"

#include "doctest/doctest.h"
#include <testHelpers.h>

using namespace elsa;
using namespace doctest;

TEST_SUITE_BEGIN("ml");

TEST_CASE_TEMPLATE("AutoEncoder: pre-training testing", TestType, float)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor mnistShape({28, 28, 1});
        index_t batchSize = 32;

        WHEN("instantiating an AutoEncoder model")
        {
            THEN("no error is thrown")
            {
                REQUIRE_NOTHROW(
                    ml::AutoEncoder<TestType, elsa::ml::MlBackend::Cudnn>{mnistShape, batchSize});
            }
        }

        WHEN("instantiating an AutoEncoder model")
        {
            ml::AutoEncoder<TestType, elsa::ml::MlBackend::Cudnn> aeModel(mnistShape, batchSize);

            THEN("the shape of the input matches the shape of the output")
            {
                DataContainer<TestType> randomValues(mnistShape);
                DataContainer<TestType> prediction = aeModel.predict(randomValues);
                REQUIRE_EQ(prediction.getDataDescriptor(), mnistShape);
            }
        }
    }
}

TEST_CASE_TEMPLATE("AutoEncoder: testing of training", TestType, float)
{
    GIVEN("a DataDescriptor")
    {
        VolumeDescriptor mnistShape({28, 28, 1});
        index_t batchSize = 32;

        std::vector<DataContainer<TestType>> inputs;
        // TODO perhaps generate one phantom image and augment it
        std::vector<DataContainer<TestType>> labels;
        // TODO perhaps generate one phantom image

        WHEN("instantiating and training an AutoEncoder model")
        {
            ml::AutoEncoder<TestType, elsa::ml::MlBackend::Cudnn> aeModel(mnistShape, batchSize);

            // define an Adam optimizer
            auto optimizer = ml::Adam<TestType>();

            // compile the model
            aeModel.compile(ml::SparseCategoricalCrossentropy<TestType>(), &optimizer);

            typename ml::Model<TestType, ml::MlBackend::Cudnn>::History history =
                aeModel.fit(inputs, labels, 2);

            THEN("all losses are non-NaNs")
            {
                std::vector<TestType> losses = history.loss;
                for (auto loss : losses) {
                    REQUIRE_UNARY(!std::isnan(loss));
                }
            }

            THEN("no exception is thrown when predicting")
            {
                DataContainer<TestType> randomValues(mnistShape);
                REQUIRE_NOTHROW(aeModel.predict(randomValues));
            }

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
