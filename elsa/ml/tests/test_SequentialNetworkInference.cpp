#include <catch2/catch.hpp>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "DataContainer.h"
#include "SequentialNetwork.h"

using namespace elsa;

TEST_CASE("SequentialNetwork Inference", "elsa_ml")
{
    IndexVector_t inputVec(2);
    // One batch dimension and input size is 5
    inputVec << 2, 5;
    DataDescriptor inputDesc(inputVec);

    Eigen::VectorXf inputValues = Eigen::VectorXf::Random(2 * 5);

    DataContainer<float> input(inputDesc, inputValues);

    auto model = SequentialNetwork<float, MlBackend::Dnnl>(inputDesc);

    // Add network layers
    model
        // Dense layer with 3 neurons and all weights/biases are 1
        .addDenseLayer(3, Initializer::One)
        // Relu layer with negative slope 0.5
        .addActivationLayer(Activation::Relu, .5f)
        // // Dense layer with 5 neurons and all weights/biases are 1
        .addDenseLayer(5, Initializer::One)
        // Linear activation layer
        .addActivationLayer(Activation::Linear, .65f, .123f);

    model.compile();

    // First forward propagation
    model.forwardPropagate(input);
    auto output = model.getOutput();

    // Calculate expected output
    Eigen::MatrixXf weightsRequired1 = Eigen::MatrixXf::Ones(3, 5);
    Eigen::MatrixXf weightsRequired2 = Eigen::MatrixXf::Ones(5, 3);
    Eigen::VectorXf biasRequired1 = Eigen::VectorXf::Ones(3);
    Eigen::VectorXf biasRequired2 = Eigen::VectorXf::Ones(5);

    Eigen::VectorXf firstBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.head(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    Eigen::VectorXf secondBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.tail(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    for (int i = 0; i < 5; ++i)
        REQUIRE(firstBatchRequired[i] == Approx(output[i]));

    for (int i = 5; i < 10; ++i)
        REQUIRE(secondBatchRequired[i - 5] == Approx(output[i]));

    // Second forward propagation
    inputValues.setRandom(2 * 5);
    input = DataContainer<float>(inputDesc, inputValues);

    model.forwardPropagate(input);
    output = model.getOutput();

    // Calculate expected output
    firstBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.head(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    secondBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.tail(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    for (int i = 0; i < 5; ++i)
        REQUIRE(firstBatchRequired[i] == Approx(output[i]));

    for (int i = 5; i < 10; ++i)
        REQUIRE(secondBatchRequired[i - 5] == Approx(output[i]));

    // Third forward propagation
    inputValues.setRandom(2 * 5);
    input = DataContainer<float>(inputDesc, inputValues);

    model.forwardPropagate(input);
    output = model.getOutput();

    // Calculate expected output
    firstBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.head(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    secondBatchRequired =
        (weightsRequired2
             * (weightsRequired1 * inputValues.tail(5) + biasRequired1).unaryExpr([](float coeff) {
                   return (coeff < 0) ? (.5f * coeff) : (coeff);
               })
         + biasRequired2)
            .unaryExpr([](float coeff) { return .65f * coeff + .123f; });

    for (int i = 0; i < 5; ++i)
        REQUIRE(firstBatchRequired[i] == Approx(output[i]));

    for (int i = 5; i < 10; ++i)
        REQUIRE(secondBatchRequired[i - 5] == Approx(output[i]));
}

TEST_CASE("SequentialNetwork Failtest", "elsa_ml")
{
    IndexVector_t inputVec(2);
    // One batch dimension and input size is 5
    inputVec << 2, 5;
    DataDescriptor inputDesc(inputVec);

    Eigen::VectorXf inputValues = Eigen::VectorXf::Random(2 * 5);

    DataContainer<float> input(inputDesc, inputValues);

    auto model = SequentialNetwork<float, MlBackend::Dnnl>(inputDesc);

    // Compilation of empty network should throw
    REQUIRE_THROWS_WITH(model.compile(), "Cannot compile network: Network contains not layers");

    // Empty network should throw when asked for output descriptor or output
    REQUIRE_THROWS_WITH(model.getOutputDescriptor(),
                     "Cannot return network output descriptor because network contains no layers");
    REQUIRE_THROWS_WITH(model.getOutput(),
                     "Cannot get network output: The network contains not layers");

    // Add network layers
    model
        // Dense layer with 3 neurons and all weights/biases are 1
        .addDenseLayer(3, Initializer::One)
        // Relu layer with negative slope 0.5
        .addActivationLayer(Activation::Relu, .5f)
        // // Dense layer with 5 neurons and all weights/biases are 1
        .addDenseLayer(5, Initializer::One)
        // Linear activation layer
        .addActivationLayer(Activation::Linear, .65f, .123f);

    // Cannot get output without prior propagation
    REQUIRE_THROWS_WITH(model.getOutput(), "Cannot get network output: No input has been propagated");

    // Forward propagation without previous network compilation should throw
    REQUIRE_THROWS_WITH(model.forwardPropagate(input), Catch::Contains("Use SequentialNetwork::compile"));
}