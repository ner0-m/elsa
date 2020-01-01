#include <catch2/catch.hpp>
#include <type_traits>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "Dense.h"

using namespace elsa;

TEST_CASE("Dense semantics")
{
    SECTION("Test 1")
    {
        IndexVector_t inputVec(2);
        // One batch dimension and input size is 5
        inputVec << 1, 5;
        DataDescriptor inputDesc(inputVec);
        Eigen::VectorXf inputValues(1 * 5);
        inputValues << 1, 2, 1, 1, 5;
        DataContainer<float> input(inputDesc, inputValues);

        IndexVector_t weightsVec(2);
        // Three neurons and input size is 5
        weightsVec << 3, 5;
        DataDescriptor weightsDesc(weightsVec);
        Eigen::VectorXf weightsValues(3 * 5);
        // clang-format off
        weightsValues <<  1,  2,  3,  4,  5,
                         -1, -2, -3, -4, -5,
                          5,  4,  3,  2,  1;
        // clang-format on
        DataContainer<float> weights(weightsDesc, weightsValues);

        IndexVector_t biasVec(1);
        biasVec << 3;
        DataDescriptor biasDesc(biasVec);
        Eigen::VectorXf biasValues(1 * 3);
        biasValues << 1, 0, 1;
        DataContainer<float> bias(biasDesc, biasValues);

        Dense<float> dense(inputDesc, 3);

        auto backend = dense.getBackend();
        backend->setInput(input);
        std::static_pointer_cast<typename Dense<float>::BackendLayerType>(backend)->setWeights(
            weights);
        std::static_pointer_cast<typename Dense<float>::BackendLayerType>(backend)->setBias(bias);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        Eigen::VectorXf required(3);
        required << 38, -37, 24;

        for (int i = 0; i < 3; ++i)
            REQUIRE(output[i] == required[i]);
    }
}