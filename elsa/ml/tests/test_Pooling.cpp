#include <catch2/catch.hpp>
#include <type_traits>
#include <random>

#include "elsaDefines.h"
#include "DataDescriptor.h"
#include "PoolingLayer.h"

using namespace elsa;

TEST_CASE("Pooling semantics", "elsa_ml")
{
    SECTION("Forward")
    {
        // Example from http://cs231n.github.io/convolutional-networks/
        IndexVector_t inputVec(4);
        inputVec << 1, 1, 4, 4;
        DataDescriptor inputDesc(inputVec);

        IndexVector_t stridesVec(2);
        stridesVec << 2, 2;

        IndexVector_t poolingVec(2);
        poolingVec << 2, 2;

        PoolingLayer<float> pool(inputDesc, poolingVec, stridesVec);

        Eigen::VectorXf vec(1 * 1 * 4 * 4);
        // clang-format off
        vec << 1, 1, 2, 4,
               5, 6, 7, 8,
               3, 2, 1, 0,
               1, 2, 3, 4;
        // clang-format on

        DataContainer<float> input(inputDesc, vec);

        auto backend = pool.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        Eigen::VectorXf required(1 * 1 * 2 * 2);
        // clang-format off
        required << 6, 8,
                    3, 4;
        // clang-format on

        for (int i = 0; i < 4; ++i)
            REQUIRE(output[i] == required[i]);
    }

    SECTION("Backward")
    {
        IndexVector_t inputVec(4);
        inputVec << 1, 1, 4, 4;
        DataDescriptor inputDesc(inputVec);

        IndexVector_t stridesVec(2);
        stridesVec << 2, 2;

        IndexVector_t poolingVec(2);
        poolingVec << 2, 2;

        PoolingLayer<float> pool(inputDesc, poolingVec, stridesVec);

        Eigen::VectorXf vec(1 * 1 * 4 * 4);
        // clang-format off
        vec << 1, 1, 2, 4,
               5, 6, 7, 8,
               3, 2, 1, 0,
               1, 2, 3, 4;
        // clang-format on

        DataContainer<float> input(inputDesc, vec);

        auto backend = pool.getBackend();
        backend->setInput(input);
        backend->compile();
        auto engine = backend->getEngine();
        dnnl::stream s(*engine);
        backend->forwardPropagate(s);
        auto output = backend->getOutput();

        backend->setOutputGradient(output);
        backend->compile(PropagationKind::Full);
        backend->forwardPropagate(s);
        backend->backwardPropagate(s);
        auto inputGradient = backend->getInputGradient();

        auto inputGradientDesc = inputGradient.getDataDescriptor();

        Eigen::MatrixXf requiredGradientInput(4, 4);
        // clang-format off
        requiredGradientInput << 0, 0, 0, 0,
                                 0, 6, 0, 8,
                                 3, 0, 0, 0,
                                 0, 0, 0, 4;
        // clang-format on

        for (int k = 0; k < inputGradientDesc.getNumberOfCoefficientsPerDimension()[3]; ++k)
            for (int l = 0; l < inputGradientDesc.getNumberOfCoefficientsPerDimension()[2]; ++l)
                REQUIRE(inputGradient(0, 0, l, k) == Approx(requiredGradientInput(k, l)));
    }
}
