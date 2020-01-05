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

    Eigen::VectorXf inputValues(2 * 5);
    inputValues << /* First batch */ 1.f, -2.f, 1.f, 1.f, 5.f,
        /*  Second batch */ 1.5f, -4.f, 3.123f, 1.f, -100.f;

    DataContainer<float> input(inputDesc, inputValues);

    auto model = SequentialNetwork<float, MlBackend::Dnnl>(inputDesc);

    // Add network layers
    model
        // Dense layer with 5 neurons and all weights/biases are 1
        .addDenseLayer(3, Initializer::One)
        // Relu layer with negative slope 0.5
        .addActivationLayer(Activation::Relu, .5f)
        // Dense layer with 2 neurons and all weights/biases are 1
        .addDenseLayer(5, Initializer::One);

    model.compile();

    model.forwardPropagate(input);

    auto output = model.getOutput();

    Eigen::MatrixXf weightsRequired1 = Eigen::MatrixXf::Ones(3, 5);
    Eigen::MatrixXf weightsRequired2 = Eigen::MatrixXf::Ones(5, 3);
    Eigen::VectorXf biasRequired1 = Eigen::VectorXf::Ones(3);
    Eigen::VectorXf biasRequired2 = Eigen::VectorXf::Ones(5);

    Eigen::VectorXf firstBatchInput(5);
    firstBatchInput << 1.f, -2.f, 1.f, 1.f, 5.f;
    Eigen::VectorXf secondBatchInput(5);
    secondBatchInput << 1.5f, -4.f, 3.123f, 1.f, -100.f;
    Eigen::VectorXf firstBatchRequired =
        weightsRequired2
            * (weightsRequired1 * firstBatchInput + biasRequired1).unaryExpr([](float coeff) {
                  return (coeff < 0) ? (.5f * coeff) : (coeff);
              })
        + biasRequired2;
    Eigen::VectorXf secondBatchRequired =
        weightsRequired2
            * (weightsRequired1 * secondBatchInput + biasRequired1).unaryExpr([](float coeff) {
                  return (coeff < 0) ? (.5f * coeff) : (coeff);
              })
        + biasRequired2;

    for (int i = 0; i < 5; ++i)
        REQUIRE(firstBatchRequired[i] == output[i]);

    for (int i = 5; i < 10; ++i)
        REQUIRE(secondBatchRequired[i - 5] == output[i]);
}